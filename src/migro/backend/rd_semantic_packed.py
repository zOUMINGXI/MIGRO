from .paths import asset_path
"""A denser shared semantic coordinate code buys a smaller reranking budget."""
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch,zlib
from torch.nn import functional as F
from .rd_semantic_factor import SemanticFactorMemory,FactorConfig
@dataclass(frozen=True)
class PackedConfig(FactorConfig):
 key_mode:str="semantic4"
 candidates:int=16
 group:int=32
 rotation:str="learned"
 basis_path:str="research/rd_semantic_factor_20261025/packed/model/basis.pt"
class SemanticPackedMemory(SemanticFactorMemory):
 def __init__(self,core,cfg=PackedConfig()):
  super().__init__(core,cfg)
  d=torch.load(asset_path(cfg.basis_path),map_location=self.device,weights_only=True);self.full_basis=d["basis"].float();self.axis_scale=d.get("std",torch.ones(384,device=self.device)).float()
  if cfg.rotation=="identity":self.full_basis=torch.eye(384,device=self.device);self.axis_scale=torch.ones(384,device=self.device)
  self.keys=torch.empty(0,192,device=self.device,dtype=torch.uint8)
  self.scales=torch.empty(0,384//cfg.group,device=self.device,dtype=torch.float16)
 def write(self,address,ids,text):
  ids=ids.detach().cpu().numpy().astype(np.int64);key=F.normalize(address[:384].float().to(self.device),dim=-1)
  topic=self.choose_topic(key,ids);self.topics=torch.cat([self.topics,torch.tensor([topic],device=self.device,dtype=torch.uint8)])
  self.bindings.append(ids)
  original=self._semantic_ids(text)
  if self._semantic_ids(self.core.codec.tokenizer.decode(ids.tolist(),skip_special_tokens=True))!=original:
   self.exceptions[self.n]=zlib.compress(np.asarray(original,dtype="<i4").tobytes(),6)
  target=((key-self.mean)@self.full_basis)/self.axis_scale
  with torch.enable_grad():
   w=torch.zeros_like(target,requires_grad=True);loss=.5*(w-target).square().sum();value=(w-torch.autograd.grad(loss,w)[0]).detach()
  grouped=value.reshape(-1,self.cfg.group);scale=(grouped.abs().amax(1)/7).clamp_min(1e-7).half()
  quant=(grouped/scale.float()[:,None]).round().clamp(-7,7).to(torch.int16).flatten()+8
  packed=(quant[::2]|(quant[1::2]<<4)).to(torch.uint8)
  self.keys=torch.cat([self.keys,packed[None]]);self.scales=torch.cat([self.scales,scale[None]]);self.n+=1
  return dict(write_loss=float(loss),binding_mode=self.bindings.modes[-1],topic=topic)
 def coordinates(self):
  low=(self.keys&15).to(torch.float32)-8;high=(self.keys>>4).to(torch.float32)-8
  return (torch.stack([low,high],-1).reshape(self.n,-1,self.cfg.group)*self.scales.float()[:,:,None]).reshape(self.n,384)*self.axis_scale
 @torch.no_grad()
 def route(self,query,topk=4,candidates=None):
  q=F.normalize(self.core.encode_semantic(["Represent this sentence for searching relevant passages: "+query,query])[:,:384].float(),dim=-1)
  scores=(q@self.full_basis@self.coordinates().T+(q@self.mean)[:,None]).amax(0)
  c=self.cfg.candidates if candidates is None else candidates
  if c==0:
   self.last_route_stats=dict(reranked=0);return sorted(scores.topk(min(topk,self.n)).indices.tolist())
  slots=scores.topk(min(c,self.n)).indices.tolist();exact=self._exact_keys(slots)
  which=(q@exact.float().T).amax(0).topk(min(topk,len(slots))).indices.tolist()
  self.last_route_stats=dict(reranked=len(slots),source_history_loaded=False,values_reconstructed_from_current_state=True)
  return sorted(slots[j] for j in which)
 def metadata(self):
  d=super().metadata();d.update(schema="semantic-packed-v1",candidate_memory_reencoding=True);return d
 def ledger(self):
  d=super().ledger();d["shared_codec_file_bytes"]+=asset_path(self.cfg.basis_path).stat().st_size
  d["shared_codec_runtime_array_bytes"]+=(self.full_basis.numel()+self.axis_scale.numel())*self.full_basis.element_size();return d
