"""Actual-byte, multi-constraint allocation of shared continuous address pages.
Frozen hosts/core; exact literal bindings. This proposes allocations, not quality certificates.
"""
import json,zlib,struct,copy,math
from pathlib import Path
import numpy as np,torch
from scipy.optimize import milp,LinearConstraint,Bounds
from scipy.sparse import lil_matrix
from .rd_shared_anchors import SharedAnchorMemory,AnchorConfig,AnchorBindings
from .rd_runtime_accounting import share_public_tables
from .rd_rate_v2 import encode_blob,decode_blob
MAGIC=b"EVIRP01\0"
GROUPS=("main","old","revision","exact","abstention","answerable")
OPTIONS=("f32","f16","q8","q4","q2","lr8r8","lr4r8","lr8r16","lr4r16","lr4r32")
def bundle(meta,arrays):
 header=[];payload=[]
 for name,array in arrays.items():
  a=np.ascontiguousarray(array);header.append((name,a.dtype.str,a.shape,a.nbytes));payload.append(a.tobytes())
 h=json.dumps([meta,header],separators=(",",":")).encode()
 return zlib.compress(struct.pack("<I",len(h))+h+b"".join(payload),6)
def unbundle(blob):
 raw=zlib.decompress(blob);n=struct.unpack("<I",raw[:4])[0];meta,header=json.loads(raw[4:4+n]);off=4+n;arrays={}
 for name,dtype,shape,size in header:
  arrays[name]=np.frombuffer(raw[off:off+size],dtype=dtype).reshape(shape).copy();off+=size
 assert off==len(raw)
 return meta,arrays
def quantize(a,bits):
 a=np.asarray(a,np.float32);shape=a.shape;w=shape[-1];pad=(-w)%32
 x=np.pad(a,[(0,0)]*(a.ndim-1)+[(0,pad)]).reshape(-1,32)
 limit=2**(bits-1)-1;sc=np.maximum(np.max(np.abs(x),axis=1)/limit,1e-7).astype("<f2")
 q=np.clip(np.rint(x/sc[:,None].astype(np.float32)),-limit,limit).astype(np.int16)+2**(bits-1)
 q=q.reshape(-1,8//bits);packed=np.zeros(len(q),np.uint8)
 for i in range(8//bits):packed|=(q[:,i].astype(np.uint8)<<(i*bits))
 restored=(q.reshape(-1,32).astype(np.float32)-2**(bits-1))*sc[:,None].astype(np.float32)
 restored=restored.reshape(*shape[:-1],w+pad)[...,:w]
 return dict(q=packed,s=sc),restored,dict(shape=shape,bits=bits,pad=pad)
def dequant(arr,md,prefix=""):
 bits=md["bits"];q=arr[prefix+"q"];unpacked=np.stack([(q>>(i*bits))&((1<<bits)-1) for i in range(8//bits)],axis=1).reshape(-1,32).astype(np.float32)-2**(bits-1)
 x=unpacked*arr[prefix+"s"][:,None].astype(np.float32);shape=md["shape"]
 return x.reshape(*shape[:-1],shape[-1]+md["pad"])[...,:shape[-1]]
def address_candidates(keys):
 x=np.asarray(keys,np.float32);out=[]
 # SVD is learned solely from this completed observation page, before any future query.
 u,s,vt=np.linalg.svd(x-x.mean(0),full_matrices=False);mean=x.mean(0).astype("<f4")
 for option in OPTIONS:
  if option.startswith("f"):
   a=x.astype("<f4" if option=="f32" else "<f2");md=dict(option=option);arrays=dict(x=a);decoded=a.astype(np.float32)
  elif option.startswith("q"):
   arrays,decoded,qmeta=quantize(x,int(option[1:]));md=dict(option=option,q=qmeta)
  else:
   bits,rank=map(int,option[2:].split("r"));rank=min(rank,len(s))
   left=u[:,:rank]*s[None,:rank];right=vt[:rank]
   la,ld,lm=quantize(left,bits);ra,rd,rm=quantize(right,bits)
   arrays={**{"l"+k:v for k,v in la.items()},**{"r"+k:v for k,v in ra.items()},"mean":mean}
   decoded=ld@rd+mean;md=dict(option=option,l=lm,r=rm)
  blob=bundle(md,arrays);out.append(dict(option=option,blob=blob,decoded=decoded,bytes=len(blob)))
 return out
def decode_address(blob):
 md,a=unbundle(blob);option=md["option"]
 if option.startswith("f"):return a["x"].astype(np.float32)
 if option.startswith("q"):return dequant(a,md["q"])
 return dequant(a,md["l"],"l")@dequant(a,md["r"],"r")+a["mean"]
def binding_owner(core):
 return share_public_tables(SharedAnchorMemory(core,AnchorConfig(share_keys=False,share_binding=False)))
def pack_bindings(owner,ids):
 bindings=AnchorBindings(owner)
 for x in ids:bindings.append(np.asarray(x,dtype=np.int64))
 return encode_blob(dict(n=len(ids)),bindings.tensors())
def unpack_bindings(owner,blob):
 md,t=decode_blob(blob,"cpu");b=AnchorBindings(owner);b.restore(t);assert len(b.lengths)==md["n"];return b
def allocate(page_options,risks,limits,time_limit=30):
 """Finite-grid mixed-integer optimum in the sum of ACTUAL coded page bytes."""
 P=len(page_options);K=len(page_options[0]);G=len(limits)
 assert risks.shape==(P,K,G)
 c=np.array([o["bytes"] for page in page_options for o in page],float)
 a=lil_matrix((P+G,P*K),dtype=np.float64)
 for p in range(P):a[p,p*K:(p+1)*K]=1
 for g in range(G):a[P+g,:]=risks[:,:,g].reshape(-1)/max(float(limits[g]),1e-15)
 lb=np.r_[np.ones(P),np.full(G,-np.inf)];ub=np.r_[np.ones(P),np.ones(G)]
 result=milp(c,integrality=np.ones(P*K),bounds=Bounds(0,1),constraints=LinearConstraint(a.tocsr(),lb,ub),options=dict(time_limit=time_limit,mip_rel_gap=.0001))
 selected=np.zeros(P,dtype=int) if result.x is None else result.x.reshape(P,K).argmax(1)
 actual=risks[np.arange(P),selected].sum(0)
 if np.any(actual>limits+1e-10):
  selected=np.zeros(P,dtype=int);actual=risks[np.arange(P),selected].sum(0)
 assert np.all(actual<=limits+1e-10)
 return selected.tolist(),dict(proxy_constraints_feasible=True,solver_status=int(result.status),finite_candidate_optimal=result.status==0,mip_gap=float(getattr(result,"mip_gap",np.nan)),actual_address_bytes=int(sum(page_options[p][k]["bytes"] for p,k in enumerate(selected))),proxy_risk=actual.tolist(),proxy_limits=np.asarray(limits).tolist(),quality_certified=False)
def risk_matrix(pages,profiles,masks):
 """Query-projection distortion per host/stratum; final answer NI is a separate gate."""
 profiles=np.asarray(profiles,np.float64);assert profiles.shape==(18,384)
 masks=np.asarray(masks,float);assert masks.shape[1]==6
 denom=np.maximum(masks.sum(0),1);start=0;rows=[]
 for options in pages:
  reference=options[0]["decoded"];n=len(reference);mask=masks[start:start+n];start+=n
  r=reference/np.maximum(np.linalg.norm(reference,axis=1,keepdims=True),1e-8);loss=[]
  for o in options:
   x=o["decoded"];x=x/np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-8)
   e=(x-r)**2;weighted=e@profiles.T
   loss.append((weighted*np.tile(mask,(1,3))).sum(0)/np.tile(denom,3))
  rows.append(loss)
 return np.asarray(rows)
class PageMemory:
 def __init__(self,core,pages):
  self.core=core;self.pages=pages;self.owner=binding_owner(core);self.bindings=[unpack_bindings(self.owner,p["bindings"]) for p in pages]
  self.ends=np.cumsum([0]+[len(b.lengths) for b in self.bindings]);self.n=int(self.ends[-1])
  self.full_basis=torch.eye(384,device=core.device);self.mean=torch.zeros(384,device=core.device)
 def coordinates(self):return torch.tensor(np.concatenate([decode_address(p["address"]) for p in self.pages]),device=self.core.device)
 def read_slot(self,i,*args):
  j=int(np.searchsorted(self.ends,i,side="right")-1)
  return torch.tensor(self.bindings[j].decode(i-self.ends[j]),device=self.core.device)
class PageState:
 def __init__(self,core,meta,events,pages):
  self.core=core;self.meta=meta;self.events=events;self.pages=pages;self.mem=PageMemory(core,pages);self.key_mode="content"
  assert len(meta)==self.mem.n
 def serialize(self):
  header=zlib.compress(json.dumps(dict(schema="robust-address-pages-v1",meta=self.meta,events=self.events),separators=(",",":")).encode(),6)
  parts=[MAGIC,struct.pack("<II",len(header),len(self.pages)),header]
  for p in self.pages:parts.extend([struct.pack("<II",len(p["address"]),len(p["bindings"])),p["address"],p["bindings"]])
  return b"".join(parts)
 @classmethod
 def restore(cls,core,blob):
  assert blob[:8]==MAGIC
  h,n=struct.unpack("<II",blob[8:16]);d=json.loads(zlib.decompress(blob[16:16+h]));off=16+h;pages=[]
  for i in range(n):
   a,b=struct.unpack("<II",blob[off:off+8]);off+=8;pages.append(dict(address=blob[off:off+a],bindings=blob[off+a:off+a+b]));off+=a+b
  assert off==len(blob)
  return cls(core,d["meta"],d["events"],pages)
 def handoff(self):return self.restore(self.core,self.serialize())
def restore_any(core,blob):
 if blob[:8]==MAGIC:return PageState.restore(core,blob)
 from .rd_elastic_anchors import restore_dialogue
 return restore_dialogue(core,blob)
def prepare_pages(core,state,page_size=128):
 raw=(state.mem.coordinates()@state.mem.full_basis.T+state.mem.mean).detach().cpu().numpy();owner=binding_owner(core);pages=[];bindings=[]
 for st in range(0,len(raw),page_size):
  pages.append(address_candidates(raw[st:st+page_size]))
  bindings.append(pack_bindings(owner,[state.mem.read_slot(i,False).cpu().numpy() for i in range(st,min(st+page_size,len(raw)))]))
 return pages,bindings
def realize(core,state,pages,bindings,choice):
 result=PageState(core,copy.deepcopy(state.meta),copy.deepcopy(state.events),[dict(address=p[k]["blob"],bindings=b) for p,b,k in zip(pages,bindings,choice)])
 # The choice objective is exactly additive because all remaining bytes are common.
 extra=len(result.serialize())-sum(p[k]["bytes"] for p,k in zip(pages,choice))
 assert extra==len(realize_header(state))+sum(len(b)+8 for b in bindings)
 return result
def realize_header(state):
 h=zlib.compress(json.dumps(dict(schema="robust-address-pages-v1",meta=state.meta,events=state.events),separators=(",",":")).encode(),6)
 return MAGIC+struct.pack("<II",len(h),0)+h
