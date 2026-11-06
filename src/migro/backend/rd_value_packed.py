"""Lossless cross-observation value packing; geometry and bindings remain exact.

This is a codec engineering improvement, not learned semantic compression.
All competing representations receive the same candidate codec choices.
"""
import copy,json,struct,zlib,lzma
import numpy as np
import torch
from .rd_dialogue import DialogueState
from .rd_shared_anchors import AnchorBindings
from .rd_robust_residual import pack_storage,unpack_storage,restore_any
MAGIC=b"EVIVAL01"
class ArrayBindings:
 def __init__(self,flat,lengths):
  self.flat=flat;self.lengths=lengths;self.modes=[6];self.max_scratch=flat.nbytes;self.ends=np.r_[0,np.cumsum(lengths,dtype=np.int64)]
 def decode(self,i):return self.flat[self.ends[i]:self.ends[i+1]].astype(np.int64)
 def append(self,ids):
  ids=np.asarray(ids,np.int64)
  if len(ids) and (ids.min()<0 or ids.max()>65535):raise ValueError("Token outside packed uint16 vocabulary")
  self.flat=np.r_[self.flat,ids.astype("<u2")];self.lengths=np.r_[self.lengths,np.uint32(len(ids))];self.ends=np.r_[self.ends,self.ends[-1]+len(ids)]
def geometry_blob(state):
 base=copy.copy(state);base.mem=copy.copy(state.mem);base.mem.bindings=AnchorBindings(base.mem)
 # Explicitly use the normal state serializer with an empty value payload.
 return DialogueState.serialize(base)
def candidate_blobs(state):
 n=state.mem.n
 values=[np.asarray(state.mem.read_slot(i,False).cpu(),dtype=np.int64) for i in range(n)]
 if any(len(v) and (v.min()<0 or v.max()>65535) for v in values):raise ValueError("Packed vocabulary overflow")
 lengths=np.asarray([len(x) for x in values],dtype="<u4")
 flat=np.concatenate(values).astype("<u2") if values else np.empty(0,dtype="<u2")
 geo=geometry_blob(state);candidates={}
 # Prefix payload includes lengths, so every observation can be restored exactly.
 payload=lengths.tobytes()+flat.tobytes()
 for code,compress in [("z",lambda b:zlib.compress(b,9)),("x",lambda b:lzma.compress(b,preset=6))]:
  for layout in ["tokens","byte_planes"]:
   ordered=payload if layout=="tokens" else lengths.tobytes()+flat.view(np.uint8).reshape(-1,2).T.copy().tobytes()
   header=json.dumps(dict(n=n,codec=code,layout=layout),separators=(",",":")).encode()
   blob=MAGIC+struct.pack("<II",len(header),len(geo))+header+geo+compress(ordered)
   candidates[code+"_"+layout]=blob
 return candidates
class ValuePackedState(DialogueState):
 def __init__(self,base):self.__dict__.update(base.__dict__)
 def serialize(self):
  candidates=candidate_blobs(self);return min(candidates.values(),key=len)
 def handoff(self,target=None):return self.restore(self.core,self.serialize())
 @classmethod
 def restore(cls,core,blob):
  assert blob[:8]==MAGIC
  h,g=struct.unpack("<II",blob[8:16]);md=json.loads(blob[16:16+h]);geo=blob[16+h:16+h+g]
  base=restore_any(core,geo);payload=blob[16+h+g:]
  raw=zlib.decompress(payload) if md["codec"]=="z" else lzma.decompress(payload)
  lengths=np.frombuffer(raw[:4*md["n"]],dtype="<u4").copy();values=raw[4*md["n"]:]
  if md["layout"]=="byte_planes":values=np.frombuffer(values,np.uint8).reshape(2,-1).T.copy().tobytes()
  flat=np.frombuffer(values,dtype="<u2").copy();assert sum(lengths)==len(flat) and md["n"]==base.mem.n
  base.mem.bindings=ArrayBindings(flat,lengths);return cls(base)
def storage_container(blob):
 # Apply the same optional outer codec before and after restoring a value snapshot.
 payload=unpack_storage(blob)
 packed=pack_storage(payload)
 return packed if len(packed)<len(payload) else payload
def select(state):
 values=candidate_blobs(state)
 # Packed-state serialize() selects these same four candidates; do not compress twice.
 original=min(values.values(),key=len) if isinstance(state,ValuePackedState) else state.serialize()
 candidates={"original":storage_container(original),**{k:storage_container(v) for k,v in values.items()}}
 method=min(candidates,key=lambda k:len(candidates[k]))
 return candidates[method],dict(selected=method,candidate_bytes={k:len(v) for k,v in candidates.items()})
def restore(core,blob):
 return ValuePackedState.restore(core,blob) if blob.startswith(MAGIC) else restore_any(core,blob)
def verify(old,new):
 assert old.meta==new.meta and old.events==new.events and old.key_mode==new.key_mode
 assert torch.equal(old.mem.coordinates(),new.mem.coordinates())
 assert torch.equal(old.mem.full_basis,new.mem.full_basis) and torch.equal(old.mem.mean,new.mem.mean)
 assert old.mem.n==new.mem.n
 for i in range(old.mem.n):assert torch.equal(old.mem.read_slot(i,False),new.mem.read_slot(i,False))
 return dict(all_bindings_exact=True,all_geometry_exact=True,all_metadata_exact=True,observation_count=old.mem.n)
def prefix(state,n):
 assert not state.events,"Natural prefix profiling requires no later-event metadata"
 out=DialogueState(state.core,state.key_mode);out.meta=copy.deepcopy(state.meta[:n]);out.mem=copy.copy(state.mem)
 m=out.mem;nr=sum(x<n for x in m.anchor_slots) if m.cfg.share_keys else n
 ni=sum(x<0 for x in m.references[:n]) if m.cfg.share_keys else 0
 m.n=n;m.keys=m.keys[:nr].clone();m.scales=m.scales[:nr].clone();m.topics=m.topics[:n].clone()
 m.references=m.references[:n];m.anchor_slots=m.anchor_slots[:nr];m.innovation_anchors=m.innovation_anchors[:ni]
 m.innovations=m.innovations[:ni].clone();m.innovation_scales=m.innovation_scales[:ni].clone();m.exceptions={k:v for k,v in m.exceptions.items() if k<n}
 m.bindings=AnchorBindings(m)
 for i in range(n):m.bindings.append(state.mem.read_slot(i,False).cpu().numpy())
 return out
