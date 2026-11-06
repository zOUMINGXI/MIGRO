"""Lossless cross-observation semantic prediction of address roots.

Content, original address precision, metadata and every version remain unchanged.
Independent, chronological and random predictors use the same codec search.
"""
import json,struct,zlib,lzma
import numpy as np
import torch
from .rd_rate_v2 import encode_blob,decode_blob
from .rd_robust_residual import unpack_storage
from .rd_value_packed import storage_container
MAGIC=b"EVIGEO01"

def _split(blob: bytes):
 raw=unpack_storage(blob)
 if raw[:8] not in (b"EVIVAL01",b"EVIENV01"):raise ValueError("Expected packed value or factored-envelope snapshot")
 h,g=struct.unpack("<II",raw[8:16])
 geo=raw[16+h:16+h+g];n=struct.unpack("<Q",geo[:8])[0]
 meta,t=decode_blob(geo[8+n:],"cpu")
 return raw,h,g,geo[:8+n],meta,t

def transform(blob: bytes, coordinates: np.ndarray, mode: str) -> tuple[bytes,dict]:
 raw,h,g,prefix,meta,t=_split(blob)
 old=t["compressed_keys"].numpy().tobytes();flat=np.frombuffer(zlib.decompress(old[1:]),np.uint8)
 keys=flat.reshape(192,-1).T.copy() if old[0] else flat.reshape(-1,192).copy()
 n=len(keys);q=np.stack([keys&15,keys>>4],axis=-1).reshape(n,384).astype(np.int16)
 parent=np.full(n,-1,dtype="<i4")
 if mode=="semantic":
  v=np.asarray(coordinates,np.float32);v=v/np.maximum(np.linalg.norm(v,axis=1,keepdims=True),1e-12)
  for i in range(1,n):parent[i]=int((v[:i]@v[i]).argmax())
 elif mode=="chronological":parent[1:]=np.arange(n-1)
 elif mode=="random":
  rng=np.random.default_rng(2634)
  for i in range(1,n):parent[i]=rng.integers(0,i)
 elif mode!="independent":raise ValueError(mode)
 delta=q.copy()
 for i,p in enumerate(parent):
  if p>=0:delta[i]=(q[i]-q[p])&15
 packed=(delta[:,::2]|(delta[:,1::2]<<4)).astype(np.uint8)
 arrays={"packed_row":packed,"packed_axis":packed.T.copy(),"nibbles_row":delta.astype(np.uint8),"nibbles_axis":delta.astype(np.uint8).T.copy()}
 choices={}
 for layout,a in arrays.items():
  value=(parent.tobytes() if mode!="independent" else b"")+a.tobytes()
  for codec,compress in [("z",lambda b:zlib.compress(b,9)),("x",lambda b:lzma.compress(b,preset=6))]:
   choices[layout+"_"+codec]=compress(value)
 selected=min(choices,key=lambda k:len(choices[k]))
 desc=dict(n=n,mode=mode,layout=selected.rsplit("_",1)[0],codec=selected.rsplit("_",1)[1],original_layout=int(old[0]))
 header=json.dumps(desc,separators=(",",":")).encode();payload=struct.pack("<I",len(header))+header+choices[selected]
 changed={name:(torch.from_numpy(np.frombuffer(payload,np.uint8).copy()) if name=="compressed_keys" else v) for name,v in t.items()}
 geometry=prefix+encode_blob(meta,changed)
 inner=raw[:8]+struct.pack("<II",h,len(geometry))+raw[16:16+h]+geometry+raw[16+h+g:]
 result=storage_container(MAGIC+inner)
 return result,dict(mode=mode,roots=n,old_root_code_bytes=len(old),new_root_code_bytes=len(payload),predictor_parent_bytes=0 if mode=="independent" else parent.nbytes,
  selected=selected,all_codec_bytes={k:len(v) for k,v in choices.items()},actual_container_bytes=len(result))

def expand(blob: bytes) -> bytes:
 raw=unpack_storage(blob)
 if raw[:8]!=MAGIC:raise ValueError("Unexpected root predictor container")
 raw,h,g,prefix,meta,t=_split(raw[8:])
 payload=t["compressed_keys"].numpy().tobytes();l=struct.unpack("<I",payload[:4])[0];md=json.loads(payload[4:4+l])
 coded=payload[4+l:];data=zlib.decompress(coded) if md["codec"]=="z" else lzma.decompress(coded)
 n=md["n"]
 if md["mode"]=="independent":parent=np.full(n,-1)
 else:parent=np.frombuffer(data[:4*n],dtype="<i4");data=data[4*n:]
 values=np.frombuffer(data,np.uint8)
 if md["layout"].startswith("packed"):
  a=values.reshape(192,n).T if md["layout"]=="packed_axis" else values.reshape(n,192)
  delta=np.stack([a&15,a>>4],axis=-1).reshape(n,384).astype(np.int16)
 else:delta=(values.reshape(384,n).T if md["layout"]=="nibbles_axis" else values.reshape(n,384)).astype(np.int16)
 for i,p in enumerate(parent):
  if p>=0:
   assert p<i
   delta[i]=(delta[i]+delta[p])&15
 keys=(delta[:,::2]|(delta[:,1::2]<<4)).astype(np.uint8)
 old=bytes([md["original_layout"]])+zlib.compress(keys.T.copy().tobytes() if md["original_layout"] else keys.tobytes(),6)
 t["compressed_keys"]=torch.from_numpy(np.frombuffer(old,np.uint8).copy())
 geo=prefix+encode_blob(meta,t)
 return raw[:8]+struct.pack("<II",h,len(geo))+raw[16:16+h]+geo+raw[16+h+g:]
