from .paths import asset_path
"""Causal variable-rate semantic anchors. Exact token residuals remain charged.
The objective is a query-weighted address-distortion surrogate, not task F1.
"""
import json,struct,zlib
from pathlib import Path
from dataclasses import dataclass,asdict
from array import array
import numpy as np,torch
from torch.nn import functional as F
from .rd_shared_anchors import SharedAnchorMemory,AnchorConfig
from .rd_semantic_factor import SemanticFactorMemory
from .rd_rate_v2 import encode_blob,decode_blob
from .rd_dialogue import DialogueState
from .rd_byte_accounting import deep_bytes
from .rd_runtime_accounting import share_public_tables
@dataclass(frozen=True)
class ElasticConfig(AnchorConfig):
 rd_lambda:float=.0001
 anchor_candidates:int=4
 weights_path:str=""
 sparse_block:int=8
class ElasticAnchorMemory(SharedAnchorMemory):
 def __init__(self,core,cfg=ElasticConfig()):
  super().__init__(core,cfg)
  assert cfg.sparse_block==8 and cfg.rd_lambda>=0
  self.innovation_data=bytearray();self.innovation_ends=array("I",[0])
  self.weights=torch.load(asset_path(cfg.weights_path),map_location=self.device,weights_only=True)["weights"].float() if cfg.weights_path else torch.ones(384,device=self.device)
  assert self.weights.shape==(384,) and bool((self.weights>0).all())
  share_public_tables(self)
 def _decode(self,blob):
  out=np.zeros(384,dtype=np.float32);p=0
  while p<len(blob):
   tag=blob[p];p+=1;block=tag>>2;mode=tag&3;assert block<48 and mode>0
   if mode==3:
    out[block*8:block*8+8]=np.frombuffer(blob[p:p+16],dtype="<f2").astype(np.float32);p+=16
   else:
    scale=np.frombuffer(blob[p:p+2],dtype="<f2")[0].astype(np.float32);p+=2;n=4 if mode==1 else 8
    a=np.frombuffer(blob[p:p+n],dtype=np.uint8);p+=n
    if mode==1:
     values=np.stack([a&15,a>>4],-1).reshape(-1).astype(np.int16)-8
    else:values=a.view(np.int8)
    out[block*8:block*8+8]=values.astype(np.float32)*scale
  assert p==len(blob)
  return out
 def _encode_candidate(self,residuals):
  # All candidate anchors compete using the same byte prices and distortion.
  g=residuals.reshape(-1,48,8);w=self.weights.reshape(1,48,8)
  zeros=(g.square()*w).sum(-1);costs=[zeros];values=[None];scales=[None]
  for bits,price in [(4,7),(8,11)]:
   limit=7 if bits==4 else 127
   scale=(g.abs().amax(-1)/limit).clamp_min(1e-7).half()
   quant=(g/scale.float()[...,None]).round().clamp(-limit,limit)
   restored=quant*scale.float()[...,None]
   costs.append(((restored-g).square()*w).sum(-1)+self.cfg.rd_lambda*price);values.append(quant);scales.append(scale)
  costs.append(((g.half().float()-g).square()*w).sum(-1)+self.cfg.rd_lambda*17);values.append(g.half());scales.append(None)
  costs=torch.stack(costs,-1);chosen=costs.argmin(-1);total=costs.min(-1).values.sum(-1)+self.cfg.rd_lambda*12
  winner=int(total.argmin());blob=bytearray()
  for block,mode in enumerate(chosen[winner].tolist()):
   if not mode:continue
   blob.append((block<<2)|mode)
   v=values[mode][winner,block].cpu().numpy()
   if mode==3:blob.extend(v.astype("<f2").tobytes())
   else:
    blob.extend(scales[mode][winner,block].cpu().numpy().astype("<f2").tobytes())
    if mode==1:
     a=v.astype(np.int16)+8;blob.extend((a[::2]|(a[1::2]<<4)).astype(np.uint8).tobytes())
    else:blob.extend(v.astype(np.int8).tobytes())
  return winner,bytes(blob),float(total[winner])
 def coordinates(self):
  anchors=self.anchor_coordinates()
  if self.n==0:return torch.empty(0,384,device=self.device)
  refs=np.asarray(self.references);idx=refs.copy();mask=refs<0
  if mask.any():idx[mask]=np.asarray(self.innovation_anchors)[-refs[mask]-1]
  result=anchors[torch.tensor(idx,device=self.device,dtype=torch.long)].clone()
  if mask.any():
   residual=np.stack([self._decode(self.innovation_data[a:b]) for a,b in zip(self.innovation_ends,self.innovation_ends[1:])])
   result[torch.tensor(mask,device=self.device)]+=torch.from_numpy(residual).to(self.device)
  return result
 def write(self,address,ids,text):
  token_ids=ids.detach().cpu().numpy().astype(np.int64);key=F.normalize(address[:384].float().to(self.device),dim=-1)
  v=(key-self.mean)@self.full_basis
  g=v.reshape(6,64);root_scale=(g.abs().amax(1)/7).clamp_min(1e-7).half()
  quant=(g/root_scale.float()[:,None]).round().clamp(-7,7)
  root_restored=(quant*root_scale.float()[:,None]).flatten()
  root_objective=float(((root_restored-v).square()*self.weights).sum())+self.cfg.rd_lambda*212
  shared=False;objective=root_objective
  if len(self.keys):
   anchors=self.anchor_coordinates()
   distance=((anchors-v).square()*self.weights).sum(-1)
   candidates=distance.topk(min(self.cfg.anchor_candidates,len(anchors)),largest=False).indices
   which,payload,cost=self._encode_candidate(v[None]-anchors[candidates]);anchor=int(candidates[which])
   if cost<root_objective:
    self.references.append(-len(self.innovation_anchors)-1);self.innovation_anchors.append(anchor)
    self.innovation_data.extend(payload);self.innovation_ends.append(len(self.innovation_data));shared=True;objective=cost
  if not shared:
   self.references.append(len(self.keys));self.anchor_slots.append(self.n)
   code=quant.to(torch.int16).flatten()+8
   packed=(code[::2]|(code[1::2]<<4)).to(torch.uint8)
   self.keys=torch.cat([self.keys,packed[None]]);self.scales=torch.cat([self.scales,root_scale[None]])
  self.bindings.append(token_ids)
  original=self._semantic_ids(text)
  if self._semantic_ids(self.core.codec.tokenizer.decode(token_ids.tolist(),skip_special_tokens=True))!=original:
   self.exceptions[self.n]=zlib.compress(np.asarray(original,dtype="<i4").tobytes(),6)
  self.n+=1
  return dict(shared=int(shared),rate_distortion_surrogate=objective,binding_mode=self.bindings.modes[-1])
 @torch.no_grad()
 def route(self,query,topk=4,candidates=None):
  if not self.n:return []
  q=F.normalize(self.core.encode_semantic(["Represent this sentence for searching relevant passages: "+query,query])[:,:384].float(),dim=-1)
  bank=F.normalize(self.coordinates()@self.full_basis.T+self.mean,dim=-1);scores=(q@bank.T).amax(0)
  n=self.cfg.candidates if candidates is None else candidates
  ix=scores.topk(min(n if n else topk,self.n)).indices.tolist()
  if n:
   exact=self._exact_keys(ix);pick=(q@exact.float().T).amax(0).topk(min(topk,len(ix))).indices.tolist();ix=[ix[j] for j in pick]
  self.last_route_stats=dict(reranked=len(ix) if n else 0,sparse_elastic_address=True)
  return sorted(ix)
 def tensors(self):
  t=SemanticFactorMemory.tensors(self)
  t.update(references=torch.tensor(self.references,dtype=torch.int32),anchor_slots=torch.tensor(self.anchor_slots,dtype=torch.int32),innovation_anchors=torch.tensor(self.innovation_anchors,dtype=torch.int32),
   innovation_data=torch.tensor(list(self.innovation_data),dtype=torch.uint8),innovation_ends=torch.tensor(self.innovation_ends,dtype=torch.int32))
  return t
 def metadata(self):
  return dict(schema="semantic-elastic-anchors-v1",config=asdict(self.cfg),n=self.n,representation="Causal immutable semantic anchors + sparse variable-rate innovations + lossless token bindings",
   distortion="Weighted squared PCA address error; task quality measured separately",future_query_used_for_write=False,weights_public_calibration_only=True)
 def restore(self,blob):
  meta,t=decode_blob(blob,self.device);assert meta["config"]==asdict(self.cfg);assert meta["schema"]=="semantic-elastic-anchors-v1"
  self.n=meta["n"];raw=t.pop("compressed_keys").cpu().numpy().tobytes();a=np.frombuffer(zlib.decompress(raw[1:]),dtype=np.uint8)
  a=a.reshape(192,-1).T if raw[0] else a.reshape(-1,192)
  self.keys=torch.from_numpy(a.copy()).to(self.device);self.scales=t["scales"];self.topics=t["topics"]
  for name in ("references","anchor_slots","innovation_anchors"):setattr(self,name,array("i",t[name].cpu().tolist()))
  self.innovation_data=bytearray(t["innovation_data"].cpu().numpy().tobytes());self.innovation_ends=array("I",t["innovation_ends"].cpu().tolist())
  self.bindings.restore(t);ix=t["exception_indices"].cpu().tolist();ends=t["exception_ends"].cpu().tolist();raw=t["exception_bytes"].cpu().numpy().tobytes()
  self.exceptions={i:raw[a:b] for i,a,b in zip(ix,ends,ends[1:])};return self
 def ledger(self):
  t=self.tensors();size=lambda names:sum(t[k].numel()*t[k].element_size() for k in names)
  address=["keys","scales","references","anchor_slots","innovation_anchors","innovation_data","innovation_ends"]
  tensor_bytes=sum(x.numel()*x.element_size() for x in (self.keys,self.scales))
  container_bytes=deep_bytes([self.references,self.anchor_slots,self.innovation_anchors,self.innovation_data,self.innovation_ends,self.bindings.blocks,self.bindings.modes,self.bindings.lengths,self.exceptions])
  public=sum(getattr(self,k).nbytes for k in ("global_cdf","predictions","context_tables","context_counts","offsets","cdf_offsets"))+sum(x.numel()*x.element_size() for x in (self.mean,self.full_basis,self.axis_scale,self.weights))
  return dict(serialized_bytes=len(self.serialize()),tensor_payload_bytes=size(t),directory_bytes=size(address),binding_bytes=size(["binding_bytes","binding_ends","binding_modes","lengths"]),
   exact_encoder_exception_bytes=size(["exception_bytes","exception_indices","exception_ends"]),anchors=len(self.keys),shared_observations=len(self.innovation_anchors),sparse_innovation_bytes=len(self.innovation_data),
   online_persistent_tensor_bytes=tensor_bytes,online_python_containers_and_bindings_bytes=container_bytes,online_persistent_state_bytes=tensor_bytes+container_bytes,
   shared_codec_file_bytes=asset_path(self.cfg.model_path).stat().st_size+asset_path(self.cfg.basis_path).stat().st_size+(asset_path(self.cfg.weights_path).stat().st_size if self.cfg.weights_path else 0),
   shared_codec_runtime_array_bytes=public,transient_binding_block_max_bytes=self.bindings.max_scratch,transient_dense_address_workspace_bytes=self.n*384*4)
class ElasticDialogueState(DialogueState):
 def __init__(self,core,cfg=ElasticConfig(),key_mode="content"):
  self.core=core;self.key_mode=key_mode;self.mem=ElasticAnchorMemory(core,cfg);self.meta=[];self.events=[]
 def handoff(self,target=None):
  blob=self.serialize();other=restore_dialogue(self.core,blob);assert other.serialize()==blob;return other
def restore_dialogue(core,blob):
 n=struct.unpack("<Q",blob[:8])[0];meta,_=decode_blob(blob[8+n:],"cpu")
 if meta["schema"]=="semantic-elastic-anchors-v1":return ElasticDialogueState(core,ElasticConfig(**meta["config"])).restore(blob)
 state=DialogueState(core)
 if meta["schema"]=="semantic-shared-anchors-v1":state.mem=SharedAnchorMemory(core,AnchorConfig(**meta["config"]))
 share_public_tables(state.mem)
 return state.restore(blob)
