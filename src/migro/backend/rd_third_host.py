"""Third-host interface; shared state and frozen backbones remain unchanged."""
import math
from pathlib import Path
import torch
from torch import nn
from .rd_dialogue import DialogueBridge
from .rd_benchmark import HOSTS as PREVIOUS_HOSTS
from .rd_composed_transport import ComposedTransport
from .continuous_readout import boundary_lexicon
from .paths import asset_path,model_path
HOSTS={**PREVIOUS_HOSTS,"C":model_path("MISTRAL","Mistral-7B-Instruct-v0.3")}
ROOT=Path("research/rd_third_host_frontier_20261025")
GROUP="rd-third-host-frontier-20261025"
class ResidualInterface(nn.Module):
 def __init__(self,width,rank=16):
  super().__init__();self.norm=nn.LayerNorm(width,elementwise_affine=False)
  self.down=nn.Linear(width,rank,bias=False);self.up=nn.Linear(rank,width,bias=False)
  nn.init.normal_(self.down.weight,std=1/math.sqrt(width));nn.init.zeros_(self.up.weight)
 def forward(self,x):
  x=x.float();scale=x.square().mean(-1,keepdim=True).sqrt().detach().clamp_min(1e-7)
  return x+self.up(self.down(self.norm(x)))*scale
class CalibratedTransport(nn.Module):
 def __init__(self,base,width,rank=16):
  super().__init__();self.base=base;self.adapter=ResidualInterface(width,rank)
 def forward(self,z):return self.adapter(self.base(z))
 def clear_query_cache(self):self.base.clear_query_cache()
class ThirdHostBridge(DialogueBridge):
 def __init__(self,core,host="C",checkpoint=None,interface_scope="memory"):
  self.interface_scope=interface_scope
  super().__init__(HOSTS[host],host,"concise");self.last_logits_only=True
  base=ComposedTransport(core.source.embedding.weight,self.host.get_input_embeddings(),boundary_lexicon(core.codec.tokenizer,self.tokenizer,host),core.codec.tokenizer,self.tokenizer,host).to(self.device).eval()
  base.cache_enabled=True
  self.connector=CalibratedTransport(base,self.host.config.hidden_size).to(self.device);self.connector.adapter.eval()
  if checkpoint:
   state=torch.load(asset_path(checkpoint),map_location=self.device,weights_only=True)
   self.connector.adapter.load_state_dict(state["adapter"])
   self.interface_scope=state.get("interface_scope",state.get("config",{}).get("interface_scope","memory"))
  if self.interface_scope not in ("memory","query"):raise ValueError("Unknown interface scope")
 def build_item(self,row,z=None,text=None):
  if self.interface_scope=="query" and z is not None and len(z):
   from .rd_query_interface import build_query_item
   return build_query_item(self,row,z)
  return super().build_item(row,z,text)
 def calibration_item(self,q,z,answer,native_text=None):
  x,_,meta=self.build_item(q,z=None if native_text is not None else z,text=native_text)
  ids=self.ids(answer)+[self.tokenizer.eos_token_id]
  target=torch.tensor(ids,device=self.device);emb=self.host.get_input_embeddings()(target)
  labels=torch.cat([torch.full((len(x),),-100,device=self.device,dtype=torch.long),target])
  return torch.cat([x,emb]),labels,dict(meta,answer_tokens=len(ids),sequence_tokens=len(x)+len(ids))
 def calibration_loss(self,items):
  inputs,labels=self.pack(items)
  return self.host(**inputs,labels=labels,use_cache=False).loss
