from .paths import asset_path
"""Causal shared semantic anchors with independent continuous innovations.
Anchors are immutable observations, not trainable word-table copies. Bindings
remain lossless explicit residuals; all anchor references are charged.
"""
from dataclasses import dataclass,asdict
from pathlib import Path
import ctypes,zlib,sys
from array import array
import numpy as np
import torch
from torch.nn import functional as F
from .rd_semantic_factor import SemanticFactorMemory,FactorBindings
from .rd_semantic_packed import SemanticPackedMemory,PackedConfig
from .rd_rate_v2 import encode_blob,decode_blob
from .rd_semantic_ans import encode as ans_encode,decode as ans_decode
from .rd_byte_accounting import deep_bytes
R=Path("research/rd_shared_pages_20261025")
from .paths import native_dir
lib=ctypes.CDLL(str(native_dir()/"libanchors.so"))
lib.csr_encode.argtypes=[ctypes.c_void_p,ctypes.c_int,ctypes.c_void_p,ctypes.c_int]+[ctypes.c_void_p]*6+[ctypes.c_int]
lib.csr_encode.restype=ctypes.c_int
lib.csr_decode.argtypes=[ctypes.c_void_p,ctypes.c_int,ctypes.c_int,ctypes.c_void_p,ctypes.c_int]+[ctypes.c_void_p]*6
lib.csr_decode.restype=ctypes.c_int
@dataclass(frozen=True)
class AnchorConfig(PackedConfig):
 group:int=64
 semantic_bindings:bool=False
 model_path:str=str(R/"model/csr_codec.npz")
 rank:int=64
 threshold:float=.48
 share_keys:bool=True
 share_binding:bool=True

class AnchorBindings(FactorBindings):
 def __init__(self,owner):
  super().__init__(owner);self.lengths=array("H");self.modes=array("B")
 def restore(self,t):
  super().restore(t);self.lengths=array("H",self.lengths);self.modes=array("B",self.modes)

 def enc(self,ids):
  o=self.owner;a=np.ascontiguousarray(ids,dtype=np.int64);out=np.empty(len(a)*8+16,dtype=np.uint8)
  n=lib.csr_encode(a.ctypes.data,len(a),o.global_cdf.ctypes.data,len(o.global_cdf)-1,
   o.predictions.ctypes.data,o.context_tables.ctypes.data,o.offsets.ctypes.data,o.cdf_offsets.ctypes.data,
   o.context_counts.ctypes.data,out.ctypes.data,len(out))
  if n<0:raise ValueError(n)
  return out[:n].tobytes()
 def dec(self,blob,n):
  o=self.owner;a=np.frombuffer(blob,dtype=np.uint8);out=np.empty(n,dtype=np.int64)
  code=lib.csr_decode(a.ctypes.data,len(a),n,o.global_cdf.ctypes.data,len(o.global_cdf)-1,
   o.predictions.ctypes.data,o.context_tables.ctypes.data,o.offsets.ctypes.data,o.cdf_offsets.ctypes.data,
   o.context_counts.ctypes.data,out.ctypes.data)
  if code:raise ValueError(code)
  return out
 def decode(self,i):
  c=self.owner.cfg.block;p=i//c;j=i%c;start=p*c;n=min(c,len(self.lengths)-start);b=self.blocks[p];mode=self.modes[p]
  if mode==0:
   raw=np.frombuffer(zlib.decompress(b),dtype="<u2");a=sum(self.lengths[start:i])
   return raw[a:a+self.lengths[i]].astype(np.int64)
  sizes=np.frombuffer(b[:2*n],dtype="<u2");off=3*n if mode==5 else 2*n;a=off+int(sizes[:j].sum());piece=b[a:a+int(sizes[j])]
  if mode==1:return ans_decode(piece,self.lengths[i],self.owner.global_cdf)
  if mode==3:return self.dec(piece,self.lengths[i])
  flag=b[2*n+j]
  if flag==0:return self.dec(piece,self.lengths[i])
  root=self.owner.root_slot(i);assert root<i
  dictionary=self.decode(root).astype("<u2").tobytes()
  z=zlib.decompressobj(wbits=zlib.MAX_WBITS,zdict=dictionary);raw=z.decompress(piece)+z.flush()
  return np.frombuffer(raw,dtype="<u2").astype(np.int64)
 def append(self,ids):
  c=self.owner.cfg.block;start=(len(self.lengths)//c)*c
  old=[self.decode(i) for i in range(start,len(self.lengths))];values=old+[np.asarray(ids,dtype=np.int64)]
  raw=np.concatenate(values).astype("<u2").tobytes();choices=[(zlib.compress(raw,6),0)]
  context=[self.enc(v) for v in values]
  for mode,pieces in [(1,[ans_encode(v,self.owner.global_cdf) for v in values]),(3,context)]:
   choices.append((np.asarray([len(p) for p in pieces],dtype="<u2").tobytes()+b"".join(pieces),mode))
  if self.owner.cfg.share_keys and self.owner.cfg.share_binding:
   pieces=[];flags=[]
   for j,(v,b) in enumerate(zip(values,context)):
    i=start+j;root=self.owner.root_slot(i);flag=0
    if root<i:
     rv=values[root-start] if root>=start else self.decode(root)
     z=zlib.compressobj(level=6,wbits=zlib.MAX_WBITS,zdict=rv.astype("<u2").tobytes())
     candidate=z.compress(v.astype("<u2").tobytes())+z.flush()
     if len(candidate)<len(b):b=candidate;flag=1
    pieces.append(b);flags.append(flag)
   blob=np.asarray([len(p) for p in pieces],dtype="<u2").tobytes()+bytes(flags)+b"".join(pieces)
   choices.append((blob,5))
  chosen,mode=min(choices,key=lambda p:len(p[0]))
  if len(self.lengths)%c:self.blocks[-1]=chosen;self.modes[-1]=mode
  else:self.blocks.append(chosen);self.modes.append(mode)
  self.lengths.append(len(ids));self.max_scratch=max(self.max_scratch,len(raw))

class SharedAnchorMemory(SemanticPackedMemory):
 def __init__(self,core,cfg=AnchorConfig()):
  self.core=core;self.cfg=cfg;self.device=core.device
  d=np.load(asset_path(cfg.model_path))
  for name in ("global_cdf","predictions","context_tables","context_counts","offsets","cdf_offsets"):setattr(self,name,np.ascontiguousarray(d[name]))
  self.mean=torch.from_numpy(d["mean"]).to(self.device)
  b=torch.load(asset_path(cfg.basis_path),map_location=self.device,weights_only=True)
  self.full_basis=b["basis"].float();self.axis_scale=torch.ones(384,device=self.device)
  self.keys=torch.empty(0,192,device=self.device,dtype=torch.uint8)
  self.scales=torch.empty(0,6,device=self.device,dtype=torch.float16)
  self.topics=torch.empty(0,device=self.device,dtype=torch.uint8)
  self.references=array("i");self.anchor_slots=array("i");self.innovation_anchors=array("i")
  self.innovations=torch.empty(0,cfg.rank,device=self.device,dtype=torch.int8)
  self.innovation_scales=torch.empty(0,device=self.device,dtype=torch.float16)
  self.bindings=AnchorBindings(self);self.exceptions={};self.n=0;self.last_route_stats={}
 def anchor_coordinates(self):
  n=len(self.keys)
  low=(self.keys&15).float()-8;high=(self.keys>>4).float()-8
  return (torch.stack([low,high],-1).reshape(n,6,64)*self.scales.float()[:,:,None]).reshape(n,384)
 def coordinates(self):
  a=self.anchor_coordinates()
  if not self.cfg.share_keys:return a
  refs=torch.tensor(self.references,device=self.device,dtype=torch.long);shared=refs<0
  idx=refs.clone()
  if shared.any():idx[shared]=torch.tensor(self.innovation_anchors,device=self.device)[-refs[shared]-1]
  v=a[idx]
  if shared.any():v[shared,:self.cfg.rank]+=self.innovations.float()*self.innovation_scales.float()[:,None]
  return v
 @torch.no_grad()
 def route(self,query,topk=4,candidates=None):
  if not self.n:
   self.last_route_stats=dict(reranked=0);return []
  if not self.cfg.share_keys:return SemanticPackedMemory.route(self,query,topk,candidates)
  q=F.normalize(self.core.encode_semantic(["Represent this sentence for searching relevant passages: "+query,query])[:,:384].float(),dim=-1)
  projected=q@self.full_basis
  anchors=self.anchor_coordinates()
  refs=torch.frombuffer(self.references,dtype=torch.int32).to(device=self.device,dtype=torch.long)
  mask=refs<0;indices=refs.clone()
  if len(self.innovation_anchors):
   association=torch.frombuffer(self.innovation_anchors,dtype=torch.int32).to(device=self.device,dtype=torch.long)
   indices[mask]=association[-refs[mask]-1]
  scores=(projected@anchors.T)[:,indices]
  if len(self.innovation_anchors):
   scores[:,mask]+=projected[:,:self.cfg.rank]@(self.innovations.float()*self.innovation_scales.float()[:,None]).T
  scores=(scores+(q@self.mean)[:,None]).amax(0)
  c=self.cfg.candidates if candidates is None else candidates
  if not c:
   self.last_route_stats=dict(reranked=0,factorized_address_read=True)
   return sorted(scores.topk(min(topk,self.n)).indices.tolist())
  slots=scores.topk(min(c,self.n)).indices.tolist();exact=self._exact_keys(slots)
  which=(q@exact.float().T).amax(0).topk(min(topk,len(slots))).indices.tolist()
  self.last_route_stats=dict(reranked=len(slots),factorized_address_read=True,source_history_loaded=False,values_reconstructed_from_current_state=True)
  return sorted(slots[j] for j in which)
 def root_slot(self,i):
  r=self.references[i]
  return self.anchor_slots[r if r>=0 else self.innovation_anchors[-r-1]]
 def choose_topic(self,key,ids):return 0
 def write(self,address,ids,text):
  ids=ids.detach().cpu().numpy().astype(np.int64);key=F.normalize(address[:384].float().to(self.device),dim=-1)
  v=(key-self.mean)@self.full_basis;share=False;error=0.
  if self.cfg.share_keys and len(self.keys):
   a=self.anchor_coordinates();errors=(a[:,self.cfg.rank:]-v[self.cfg.rank:]).square().sum(1)
   j=int(errors.argmin());error=float(errors[j].sqrt());share=error<=self.cfg.threshold
  if share:
   residual=v[:self.cfg.rank]-a[j,:self.cfg.rank]
   with torch.enable_grad():
    w=torch.zeros_like(residual,requires_grad=True);loss=.5*(w-residual).square().sum()
    residual=(w-torch.autograd.grad(loss,w)[0]).detach()
   scale=(residual.abs().max()/127).clamp_min(1e-7).half()
   self.references.append(-len(self.innovation_anchors)-1);self.innovation_anchors.append(j)
   self.innovations=torch.cat([self.innovations,(residual/scale.float()).round().clamp(-127,127).to(torch.int8)[None]])
   self.innovation_scales=torch.cat([self.innovation_scales,scale[None]])
  else:
   if self.cfg.share_keys:self.references.append(len(self.keys));self.anchor_slots.append(self.n)
   g=v.reshape(6,64);scale=(g.abs().amax(1)/7).clamp_min(1e-7).half()
   quant=(g/scale.float()[:,None]).round().clamp(-7,7).to(torch.int16).flatten()+8
   self.keys=torch.cat([self.keys,(quant[::2]|(quant[1::2]<<4)).to(torch.uint8)[None]])
   self.scales=torch.cat([self.scales,scale[None]])
   loss=torch.tensor(0.)
  self.bindings.append(ids)
  original=self._semantic_ids(text)
  if self._semantic_ids(self.core.codec.tokenizer.decode(ids.tolist(),skip_special_tokens=True))!=original:
   self.exceptions[self.n]=zlib.compress(np.asarray(original,dtype="<i4").tobytes(),6)
  self.n+=1
  return dict(write_loss=float(loss),shared=int(share),anchor_error=error,binding_mode=self.bindings.modes[-1])
 def tensors(self):
  t=SemanticFactorMemory.tensors(self)
  if self.cfg.share_keys:t.update(references=torch.tensor(self.references,dtype=torch.int32),anchor_slots=torch.tensor(self.anchor_slots,dtype=torch.int32),
   innovation_anchors=torch.tensor(self.innovation_anchors,dtype=torch.int32),innovations=self.innovations,innovation_scales=self.innovation_scales)
  return t
 def metadata(self):
  return dict(schema="semantic-shared-anchors-v1",config=asdict(self.cfg),n=self.n,representation="Shared semantic anchors + per-observation coefficient innovations + exact source-token binding residual",
   original_history_read=False,source_text_cache=False,public_dense_topic_tables=False)
 def serialize(self):
  t=self.tensors();k=t.pop("keys").cpu().numpy();b=min([(zlib.compress(k.tobytes(),6),0),(zlib.compress(k.T.copy().tobytes(),6),1)],key=lambda p:len(p[0]))
  t["compressed_keys"]=torch.from_numpy(np.frombuffer(bytes([b[1]])+b[0],dtype=np.uint8).copy())
  meta=self.metadata();meta["anchor_count"]=len(k)
  return encode_blob(meta,t)
 def restore(self,blob):
  meta,t=decode_blob(blob,self.device);assert meta["config"]==asdict(self.cfg);self.n=meta["n"]
  raw=t.pop("compressed_keys").cpu().numpy().tobytes();k=np.frombuffer(zlib.decompress(raw[1:]),dtype=np.uint8)
  k=k.reshape(192,-1).T if raw[0] else k.reshape(-1,192)
  self.keys=torch.from_numpy(k.copy()).to(self.device);self.scales=t["scales"];self.topics=t["topics"]
  if self.cfg.share_keys:
   for name in ("references","anchor_slots","innovation_anchors"):setattr(self,name,array("i",t[name].cpu().tolist()))
   self.innovations=t["innovations"];self.innovation_scales=t["innovation_scales"]
  else:self.references=array("i");self.anchor_slots=array("i");self.innovation_anchors=array("i")
  self.bindings.restore(t)
  ix=t["exception_indices"].cpu().tolist();ends=t["exception_ends"].cpu().tolist();raw=t["exception_bytes"].cpu().numpy().tobytes()
  self.exceptions={i:raw[a:b] for i,a,b in zip(ix,ends,ends[1:])}
  return self
 def ledger(self):
  t=self.tensors();sz=lambda names:sum(t[k].numel()*t[k].element_size() for k in names)
  public=sum(getattr(self,k).nbytes for k in ("global_cdf","predictions","context_tables","context_counts","offsets","cdf_offsets"))+sum(x.numel()*x.element_size() for x in (self.mean,self.full_basis,self.axis_scale))
  tensor_private=sum(x.numel()*x.element_size() for x in (self.keys,self.scales,self.innovations,self.innovation_scales))
  py=deep_bytes([self.references,self.anchor_slots,self.innovation_anchors,self.bindings.blocks,self.bindings.modes,self.bindings.lengths,self.exceptions])
  dirs=[k for k in t if k in ("keys","scales","references","anchor_slots","innovation_anchors","innovations","innovation_scales")]
  return dict(serialized_bytes=len(self.serialize()),tensor_payload_bytes=sz(t),directory_bytes=sz(dirs),binding_bytes=sz(["binding_bytes","binding_ends","binding_modes","lengths"]),
   exact_encoder_exception_bytes=sz(["exception_bytes","exception_indices","exception_ends"]),exceptions=len(self.exceptions),
   anchors=len(self.keys),shared_observations=len(self.innovation_anchors),binding_mode_counts={str(m):self.bindings.modes.count(m) for m in (0,1,3,5)},
   shared_codec_file_bytes=asset_path(self.cfg.model_path).stat().st_size+asset_path(self.cfg.basis_path).stat().st_size,
   shared_codec_runtime_array_bytes=public,online_persistent_tensor_bytes=tensor_private,online_python_containers_and_bindings_bytes=py,
   online_persistent_state_bytes=tensor_private+py,transient_binding_block_max_bytes=self.bindings.max_scratch)
