"""Lossless factoring of observation envelopes already present in metadata.

This is structural redundancy removal, not learned semantic abstraction.
The body tokens, temporal notes, geometry, versions and query scope stay exact.
"""
import json,struct,zlib,lzma
import numpy as np
from .rd_value_packed import geometry_blob,ArrayBindings,storage_container,ValuePackedState
from .rd_robust_residual import restore_any
MAGIC=b"EVIENV01"

def envelope_ids(core,meta):
 text="DATE: "+meta["date"]+"; "+meta["speaker"]+" said:"
 return np.asarray(core.codec.tokenizer.encode(text,add_special_tokens=False),dtype="<u2")

def candidates(state):
 flags=[];parts=[];removed=0
 for i,m in enumerate(state.meta):
  ids=np.asarray(state.mem.read_slot(i,False).cpu(),dtype=np.int64)
  if len(ids) and (ids.min()<0 or ids.max()>65535):raise ValueError("Token outside uint16")
  pre=envelope_ids(state.core,m);ok=len(pre)<=len(ids) and np.array_equal(ids[:len(pre)],pre)
  flags.append(ok);parts.append(ids[len(pre):] if ok else ids);removed+=len(pre) if ok else 0
 lengths=np.asarray([len(v) for v in parts],dtype="<u4")
 flat=np.concatenate(parts).astype("<u2") if parts else np.empty(0,dtype="<u2")
 bitmap=np.packbits(np.asarray(flags,dtype=np.uint8),bitorder="little").tobytes();geo=geometry_blob(state)
 out={}
 for codec,compress in [("z",lambda b:zlib.compress(b,9)),("x",lambda b:lzma.compress(b,preset=6))]:
  for layout in ["tokens","planes"]:
   values=flat.tobytes() if layout=="tokens" else flat.view(np.uint8).reshape(-1,2).T.copy().tobytes()
   raw=bitmap+lengths.tobytes()+values
   header=json.dumps(dict(n=state.mem.n,codec=codec,layout=layout),separators=(",",":")).encode()
   blob=MAGIC+struct.pack("<II",len(header),len(geo))+header+geo+compress(raw)
   out[codec+"_"+layout]=storage_container(blob)
 return out,dict(factored_records=sum(flags),records=len(flags),omitted_exact_envelope_tokens=removed)

def restore(core,blob):
 from .rd_robust_residual import unpack_storage
 blob=unpack_storage(blob);assert blob[:8]==MAGIC
 h,g=struct.unpack("<II",blob[8:16]);md=json.loads(blob[16:16+h]);base=restore_any(core,blob[16+h:16+h+g])
 payload=blob[16+h+g:];raw=zlib.decompress(payload) if md["codec"]=="z" else lzma.decompress(payload)
 n=md["n"];f=(n+7)//8
 flags=np.unpackbits(np.frombuffer(raw[:f],np.uint8),bitorder="little")[:n]
 lengths=np.frombuffer(raw[f:f+4*n],dtype="<u4");values=raw[f+4*n:]
 if md["layout"]=="planes":values=np.frombuffer(values,np.uint8).reshape(2,-1).T.copy().tobytes()
 flat=np.frombuffer(values,dtype="<u2");assert sum(lengths)==len(flat) and n==base.mem.n
 parts=[];st=0
 for i,length in enumerate(lengths):
  body=flat[st:st+int(length)];st+=int(length)
  parts.append(np.r_[envelope_ids(core,base.meta[i]),body] if flags[i] else body.copy())
 allids=np.concatenate(parts).astype("<u2") if parts else np.empty(0,dtype="<u2")
 base.mem.bindings=ArrayBindings(allids,np.asarray([len(v) for v in parts],dtype="<u4"))
 return ValuePackedState(base)
