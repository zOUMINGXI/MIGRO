"""Exact-rate selection; engineering fallback, not a claimed compression invention."""
from dataclasses import dataclass
MAGIC=b"EWSEL001"
QWEN_DIRECTORY_BYTES=8060917306
QWEN_NONWEIGHT_BYTES=15935306
@dataclass
class Candidate:
 name:str
 blob:bytes
 verified:bool
 def required_model_bytes(self,qwen_already_deployed):
  if qwen_already_deployed:return 0
  if self.blob.startswith(b"NCV12802"):return QWEN_DIRECTORY_BYTES
  if self.blob.startswith(b"CCM28001"):
   import struct,json
   from .causal_content_codec import decompress
   n=struct.unpack_from("<I",self.blob,8)[0];md=json.loads(decompress(self.blob[12:12+n],"zstd19"))
   if md["kind"]=="qwen_tokens":return QWEN_NONWEIGHT_BYTES
  return 0
def choose(candidates,*,qwen_already_deployed):
 valid=[c for c in candidates if c.verified]
 if not valid:raise ValueError("No exact-verified candidate")
 chosen=min(valid,key=lambda c:(len(c.blob)+c.required_model_bytes(qwen_already_deployed),c.name))
 # Only the selected stream persists. The format is already self-describing.
 blob=MAGIC+chosen.blob
 return blob,dict(selected=chosen.name,private_bytes=len(blob),
  additional_public_asset_bytes=chosen.required_model_bytes(qwen_already_deployed),
  objective_bytes=len(blob)+chosen.required_model_bytes(qwen_already_deployed),
  qwen_already_deployed=qwen_already_deployed)
def unwrap(blob):
 if not blob.startswith(MAGIC):raise ValueError("Not an exact-rate-selected archive")
 return blob[len(MAGIC):]
