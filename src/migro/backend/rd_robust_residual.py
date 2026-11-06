"""Preserve the shared semantic anchor geometry; allocate root/innovation precision.
Literal bindings, versions, ownership, and reference links stay exact and charged.
"""
import copy,json,struct,zlib
import numpy as np,torch
from .rd_robust_pages import bundle,unbundle,quantize,dequant,allocate,GROUPS
from .rd_rate_v2 import encode_blob,decode_blob
from .rd_elastic_anchors import restore_dialogue
from .rd_shared_anchors import SharedAnchorMemory,AnchorConfig
from .rd_runtime_accounting import share_public_tables
MAGIC=b"EVIRR01\0"
def decode_parameter(blob):
 md,a=unbundle(blob)
 if md["option"]=="root4":
  q=a["q"];x=np.stack([q&15,q>>4],-1).reshape(-1,6,64).astype(np.float32)-8
  return (x*a["scale"].astype(np.float32)[:,:,None]).reshape(-1,384)
 if md["option"]=="delta8":return a["q"].astype(np.float32)*a["scale"].astype(np.float32)[:,None]
 if md["option"]=="float16":return a["value"].astype(np.float32)
 return dequant(a,md["qmeta"])
def _option(md,arrays):
 b=bundle(md,arrays);return dict(option=md["option"],blob=b,bytes=len(b),decoded=decode_parameter(b))
def parameter_options(mem,page_size=128):
 roots=mem.anchor_coordinates().cpu().numpy();innov=(mem.innovations.float()*mem.innovation_scales.float()[:,None]).cpu().numpy();pages=[];layout=[]
 for kind,raw in [("root",roots),("innovation",innov)]:
  for start in range(0,len(raw),page_size):
   end=min(start+page_size,len(raw));x=raw[start:end]
   if kind=="root":exact=_option(dict(option="root4"),dict(q=mem.keys[start:end].cpu().numpy(),scale=mem.scales[start:end].cpu().numpy()));bits=[2,4]
   else:exact=_option(dict(option="delta8"),dict(q=mem.innovations[start:end].cpu().numpy(),scale=mem.innovation_scales[start:end].cpu().numpy()));bits=[2,4]
   candidates=[exact]
   for bit in bits:
    arrays,_,md=quantize(x,bit);candidates.append(_option(dict(option=f"{kind}_q{bit}",qmeta=md),arrays))
   candidates.append(_option(dict(option="float16"),dict(value=x.astype("<f2"))))
   pages.append(candidates);layout.append(dict(kind=kind,start=start,end=end))
 return pages,layout
def component_risks(mem,pages,layout,profiles,masks):
 refs=np.asarray(mem.references);assoc=np.asarray(mem.innovation_anchors);root_ids=refs.copy();shared=refs<0;root_ids[shared]=assoc[-refs[shared]-1]
 old=mem.coordinates();B=mem.full_basis;raw=old@B.T+mem.mean;oldnorm=torch.nn.functional.normalize(raw,dim=-1).cpu().numpy();basis=B.cpu().numpy()
 profiles=np.asarray(profiles,float);masks=np.asarray(masks,float);den=np.maximum(masks.sum(0),1);result=[]
 for options,desc in zip(pages,layout):
  start,end=desc["start"],desc["end"]
  if desc["kind"]=="root":
   slots=np.flatnonzero((root_ids>=start)&(root_ids<end));local=root_ids[slots]-start;width=384
  else:
   slots=np.flatnonzero(shared&((-refs-1)>=start)&((-refs-1)<end));local=-refs[slots]-1-start;width=mem.cfg.rank
  loss=[]
  for option in options:
   delta=option["decoded"]-options[0]["decoded"]
   if not len(slots):loss.append(np.zeros(18));continue
   changed=raw[slots].cpu().numpy()+delta[local]@basis[:,:width].T
   normalized=changed/np.maximum(np.linalg.norm(changed,axis=1,keepdims=True),1e-8)
   error=(normalized-oldnorm[slots])**2
   # Factor 2 conservatively budgets root and innovation contributions; this remains a proxy.
   loss.append(2*((error@profiles.T)*np.tile(masks[slots],(1,3))).sum(0)/np.tile(den,3))
  result.append(loss)
 return np.asarray(result)
def base_envelope(state):
 tensors=state.mem.tensors()
 for key in ["keys","scales","innovations","innovation_scales"]:tensors.pop(key)
 md=state.mem.metadata()
 return zlib.compress(encode_blob(dict(memory=md,meta=state.meta,events=state.events,key_mode=state.key_mode,root_count=len(state.mem.keys),innovation_count=len(state.mem.innovations)),tensors),6)
def decode_base(core,base):
 md,t=decode_blob(zlib.decompress(base),"cpu");cfg=AnchorConfig(**md["memory"]["config"]);m=share_public_tables(SharedAnchorMemory(core,cfg));m.n=md["memory"]["n"]
 from array import array
 for name in ["references","anchor_slots","innovation_anchors"]:setattr(m,name,array("i",t[name].cpu().tolist()))
 m.topics=t["topics"];m.bindings.restore(t)
 ix=t["exception_indices"].cpu().tolist();ends=t["exception_ends"].cpu().tolist();raw=t["exception_bytes"].cpu().numpy().tobytes();m.exceptions={i:raw[a:b] for i,a,b in zip(ix,ends,ends[1:])}
 return m,md
class ResidualMemory:
 def __init__(self,core,base,pages,layout):
  m,md=decode_base(core,base);cfg=m.cfg
  self.base=m;self.core=core;self.n=m.n;self.full_basis=m.full_basis;self.mean=m.mean
  roots=np.empty((md["root_count"],384),np.float32);innov=np.empty((md["innovation_count"],cfg.rank),np.float32)
  for b,d in zip(pages,layout):
   target=roots if d["kind"]=="root" else innov;target[d["start"]:d["end"]]=decode_parameter(b)
  self.roots=torch.tensor(roots,device=core.device);self.innov=torch.tensor(innov,device=core.device)
 def coordinates(self):
  refs=torch.tensor(self.base.references,device=self.core.device,dtype=torch.long);mask=refs<0;idx=refs.clone()
  if mask.any():idx[mask]=torch.tensor(self.base.innovation_anchors,device=self.core.device)[-refs[mask]-1]
  values=self.roots[idx].clone()
  if mask.any():values[mask,:self.innov.shape[1]]+=self.innov[-refs[mask]-1]
  return values
 def read_slot(self,i,*args):return self.base.read_slot(i,*args)
class ResidualState:
 def __init__(self,core,base,pages,layout):
  self.core=core;self.base_blob=base;self.page_blobs=pages;self.layout=layout
  md,_=decode_blob(zlib.decompress(base),"cpu");self.meta=md["meta"];self.events=md["events"];self.key_mode=md["key_mode"];self.mem=ResidualMemory(core,base,pages,layout)
 def serialize(self):
  header=json.dumps(self.layout,separators=(",",":")).encode();parts=[MAGIC,struct.pack("<III",len(header),len(self.base_blob),len(self.page_blobs)),header,self.base_blob]
  for b in self.page_blobs:parts.extend([struct.pack("<I",len(b)),b])
  return b"".join(parts)
 @classmethod
 def restore(cls,core,blob):
  assert blob[:8]==MAGIC
  h,b,n=struct.unpack("<III",blob[8:20]);layout=json.loads(blob[20:20+h]);base=blob[20+h:20+h+b];off=20+h+b;pages=[]
  for _ in range(n):
   size=struct.unpack("<I",blob[off:off+4])[0];off+=4;pages.append(blob[off:off+size]);off+=size
  assert off==len(blob)
  return cls(core,base,pages,layout)
 def handoff(self):return self.restore(self.core,self.serialize())
def realize(state,base,pages,layout,selected):
 return ResidualState(state.core,base,[p[k]["blob"] for p,k in zip(pages,selected)],layout)
def restore_any(core,blob):
 blob=unpack_storage(blob)
 if blob.startswith(b"DCMARC01"):
  from .decomem_archive import restore
  return restore(core,blob)
 if blob.startswith(b"EVIGEO01"):
  from .rd_semantic_root_codec import expand
  return restore_any(core,expand(blob))
 if blob.startswith(b"EVIENV01"):
  from .rd_factorized_envelope import restore
  return restore(core,blob)
 if blob.startswith(b"EVIVAL01"):
  from .rd_value_packed import ValuePackedState
  return ValuePackedState.restore(core,blob)
 if blob.startswith(b"EVEXACT1"):
  from .rd_exact_page_codec import unpack
  blob=unpack(blob)
 from .rd_scaling_reference import FloatState,MAGIC as FLOAT_MAGIC
 if blob[:8]==FLOAT_MAGIC:return FloatState.restore(core,blob)
 if blob[:8]==MAGIC:return ResidualState.restore(core,blob)
 return restore_dialogue(core,blob)

STORAGE_MAGIC=b"EVRZIP01"
def pack_storage(blob):
 compressed=zlib.compress(blob,6)
 return STORAGE_MAGIC+(b"\1"+compressed if len(compressed)<len(blob) else b"\0"+blob)
def unpack_storage(blob):
 if not blob.startswith(STORAGE_MAGIC):return blob
 return zlib.decompress(blob[9:]) if blob[8] else blob[9:]
