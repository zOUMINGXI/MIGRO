"""Lossless tensor-page alternatives; decoded old-state bytes are unchanged."""
import json,struct,zlib,lzma,base64
import numpy as np
MAGIC=b"EVEXACT1"
def raw_storage(blob):
 if blob.startswith(b"EVRZIP01"):return zlib.decompress(blob[9:]) if blob[8] else blob[9:]
 return blob
def transform(raw,mode,width):
 a=np.frombuffer(raw,np.uint8)
 if mode in (2,3):
  a=a.reshape(-1,width).copy();a[1:]=np.bitwise_xor(a[1:],a[:-1]);a=a.flatten()
 if mode in (1,3):a=a.reshape(-1,width).T.flatten()
 if mode==4:a=np.packbits(np.unpackbits(a).reshape(-1,8).T.flatten())
 return a.tobytes()
def invert(raw,mode,width):
 a=np.frombuffer(raw,np.uint8)
 if mode==4:a=np.packbits(np.unpackbits(a).reshape(8,-1).T.flatten())
 if mode in (1,3):a=a.reshape(width,-1).T.copy().flatten()
 if mode in (2,3):a=np.bitwise_xor.accumulate(a.reshape(-1,width),axis=0).flatten()
 return a.tobytes()
def choose(raw,width):
 options=[]
 for mode in range(5):
  if not raw and mode:continue
  if mode in (1,2,3) and (width<1 or len(raw)%width):continue
  data=transform(raw,mode,width)
  for codec,body in [("z",zlib.compress(data,9)),("x",lzma.compress(data,preset=6))]:
   options.append((body,dict(mode=mode,width=width,codec=codec)))
 return min(options,key=lambda x:len(x[0]))
def unpack(blob):
 if not blob.startswith(MAGIC):return raw_storage(blob)
 if blob[8:9]==b"g":return lzma.decompress(blob[9:])
 assert blob[8:9]==b"p"
 n=struct.unpack("<I",blob[9:13])[0];d=json.loads(zlib.decompress(blob[13:13+n]));off=13+n
 parts=[]
 for spec in d["parts"]:
  b=blob[off:off+spec["size"]];off+=spec["size"]
  b=zlib.decompress(b) if spec["codec"]=="z" else lzma.decompress(b)
  b=invert(b,spec["mode"],spec["width"])
  if spec.get("keys"):
   if spec["old_transposed"]:b=np.frombuffer(b,np.uint8).reshape(-1,192).T.copy().tobytes()
   b=bytes([spec["old_transposed"]])+zlib.compress(b,6)
  parts.append(b)
 assert off==len(blob)
 return base64.b64decode(d["prefix"])+base64.b64decode(d["header"])+b"".join(parts)
def pack(blob):
 raw=raw_storage(blob);options=[(blob,"original"),(MAGIC+b"g"+lzma.compress(raw,preset=6),"whole_lzma")]
 try:
  n=struct.unpack("<Q",raw[:8])[0];prefix=raw[:8+n];mem=raw[8+n:]
  if mem[:8]!=b"EVRD2001":return min(options,key=lambda x:len(x[0]))
  h=struct.unpack("<I",mem[8:12])[0];header=mem[:12+h];specs=json.loads(mem[12:12+h])["tensors"];off=12+h
  parts=[];descriptions=[]
  for spec in specs:
   chunk=mem[off:off+spec["bytes"]];off+=spec["bytes"]
   width=(spec["shape"][-1] if len(spec["shape"])>1 else 1)*np.dtype(spec["dtype"]).itemsize
   extra={}
   if spec["name"]=="compressed_keys":
    original=chunk;transposed=int(chunk[0]);expanded=zlib.decompress(chunk[1:])
    if zlib.compress(expanded,6)==chunk[1:]:
     chunk=np.frombuffer(expanded,np.uint8)
     if transposed:chunk=chunk.reshape(192,-1).T.copy()
     chunk=chunk.tobytes();width=192;extra=dict(keys=True,old_transposed=transposed)
   encoded,desc=choose(chunk,width);parts.append(encoded);descriptions.append(dict(**desc,**extra,size=len(encoded)))
  assert off==len(mem)
  description=dict(prefix=base64.b64encode(prefix).decode(),header=base64.b64encode(header).decode(),parts=descriptions)
  encoded=zlib.compress(json.dumps(description,separators=(",",":")).encode(),9)
  candidate=MAGIC+b"p"+struct.pack("<I",len(encoded))+encoded+b"".join(parts)
  assert unpack(candidate)==raw
  options.append((candidate,"tensor_pages"))
 except (AssertionError,ValueError,IndexError,struct.error,zlib.error):
  raise
 result,method=min(options,key=lambda x:len(x[0]))
 assert unpack(result)==raw
 return result,method
