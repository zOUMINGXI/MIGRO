"""Float32 address reference over the same exact observed binding content."""
import struct,zlib
import numpy as np,torch
from .rd_robust_residual import ResidualState,ResidualMemory,decode_blob,base_envelope,decode_base
MAGIC=b"EVIF3201"
class FloatMemory:
 def __init__(self,core,base,raw):
  self.base,_=decode_base(core,base);self.n=self.base.n;self.raw=raw;self.core=core
  self.full_basis=torch.eye(384,device=core.device);self.mean=torch.zeros(384,device=core.device)
 def coordinates(self):return torch.tensor(self.raw,device=self.core.device)
 def read_slot(self,i,*args):return self.base.read_slot(i,*args)
class FloatState:
 def __init__(self,core,base,raw):
  self.core=core;self.base_blob=base;self.raw=np.asarray(raw,dtype="<f4")
  md,_=decode_blob(zlib.decompress(base),"cpu");self.meta=md["meta"];self.events=md["events"];self.key_mode=md["key_mode"]
  self.mem=FloatMemory(core,base,self.raw)
 def serialize(self):return MAGIC+struct.pack("<I",len(self.base_blob))+self.base_blob+zlib.compress(self.raw.tobytes(),6)
 @classmethod
 def restore(cls,core,blob):
  assert blob[:8]==MAGIC;n=struct.unpack("<I",blob[8:12])[0]
  return cls(core,blob[12:12+n],np.frombuffer(zlib.decompress(blob[12+n:]),dtype="<f4").copy().reshape(-1,384))
