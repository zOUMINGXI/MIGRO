"""Exact content probability archive and equally charged generic controls."""
from pathlib import Path
import json,struct,subprocess,tempfile,time,sys,random,re
import numpy as np
from .paths import native_dir
import zstandard,pyppmd,brotli,zlib,lzma
BINARY=native_dir()/"causal_content"
MAGIC=b"CCM28001"
CODECS=["zstd19","ppmd8","brotli11","xz6","ppmd12"]
EXTENDED_CODECS=CODECS+["zlib9","xz9","zstd22","ppmd6","ppmd6m16","ppmd8m16"]
def compress(raw,codec):
 if codec.startswith("zstd"):return zstandard.ZstdCompressor(level=int(codec[4:])).compress(raw)
 if codec.startswith("ppmd"):
  order,_,mem=codec[4:].partition("m")
  return pyppmd.compress(raw,max_order=int(order),mem_size=int(mem or 32)<<20,variant="I")
 if codec=="brotli11":return brotli.compress(raw,quality=11)
 if codec=="zlib9":return zlib.compress(raw,9)
 return lzma.compress(raw,preset=int(codec[2:]))
def decompress(raw,codec):
 if codec.startswith("zstd"):return zstandard.ZstdDecompressor().decompress(raw)
 if codec.startswith("ppmd"):
  order,_,mem=codec[4:].partition("m")
  return pyppmd.decompress(raw,max_order=int(order),mem_size=int(mem or 32)<<20,variant="I")
 if codec=="brotli11":return brotli.decompress(raw)
 if codec=="zlib9":return zlib.decompress(raw)
 return lzma.decompress(raw)
def metadata(rows):
 # All provenance, numeric versions, timestamps and source offsets remain exact.
 return [{k:v for k,v in r.items() if k not in ["body","source_tokens"]} for r in rows]
def scopes(rows,mode):
 lookup={};out=[]
 for i,r in enumerate(rows):
  name=r.get("scope",r.get("session",""))
  if name not in lookup:lookup[name]=len(lookup)
  s=lookup[name]
  if mode=="shuffled":s=((i+1)*2654435761 & 0xffffffff)%max(1,len(lookup))
  out.append(s)
 return out
def raw_input(rows,mode):
 bodies=[r["body"].encode() for r in rows];sid=scopes(rows,mode)
 return struct.pack("<I",len(rows))+b"".join(struct.pack("<II",s,len(b)) for s,b in zip(sid,bodies))+b"".join(bodies)
def pack(meta,payload,kind,codec="",bits=18):
 md=json.dumps(dict(metadata=meta,kind=kind,codec=codec,bits=bits),ensure_ascii=False,separators=(",",":")).encode()
 header=compress(md,"zstd19")
 return MAGIC+struct.pack("<I",len(header))+header+payload
def candidate(rows,mode="scoped",bits=18,temporary_root=None):
 started=time.monotonic();raw=raw_input(rows,mode);n=len(rows);hlen=4+8*n
 with tempfile.TemporaryDirectory(prefix="ccm-",dir=temporary_root) as d:
  d=Path(d);(d/"input").write_bytes(raw)
  subprocess.run([str(BINARY),"encode",str(d/"input"),str(d/"coded"),str(bits),str(int(mode!="global"))],check=True)
  coded=(d/"coded").read_bytes()
  # Row lengths and causal namespace IDs are charged in the compressed header.
  route=compress(coded[:hlen],"zstd19")
  payload=struct.pack("<I",len(route))+route+coded[hlen:]
 blob=pack(metadata(rows),payload,mode,bits=bits)
 return blob,dict(encode_seconds=time.monotonic()-started,content_model_table_RAM_bytes=12*(1<<bits)*4,
  public_model_weights_bytes=0,compiler_binary_bytes=BINARY.stat().st_size)
def literal_variants(rows,tokenizer=None,qwen_tokenizer=None):
 bodies=[r["body"].encode() for r in rows]
 raw=struct.pack("<I",len(rows))+b"".join(struct.pack("<I",len(x)) for x in bodies)+b"".join(bodies)
 yield "utf8",raw
 for kind,tok,size in [("tokens",tokenizer,2),("qwen_tokens",qwen_tokenizer,4)]:
  if tok is None:continue
  parts=[]
  batches=tok([r["body"] for r in rows],add_special_tokens=False)["input_ids"]
  for r,ids in zip(rows,batches):
   canon=tok.decode(ids,skip_special_tokens=False);escape=b""
   if canon!=r["body"]:
    z=zlib.compressobj(9,zdict=canon.encode()[-32768:]);escape=z.compress(r["body"].encode())+z.flush()
   parts.append(struct.pack("<II",len(ids),len(escape))+np.asarray(ids,"<u2" if size==2 else "<u4").tobytes()+escape)
  yield kind,struct.pack("<I",len(parts))+b"".join(parts)

def optimized_literal(rows,tokenizer=None,codecs=CODECS,qwen_tokenizer=None):
 start=time.monotonic();options=[];best=None
 md=metadata(rows)
 for kind,raw in literal_variants(rows,tokenizer,qwen_tokenizer):
  for codec in codecs:
   t=time.monotonic();payload=compress(raw,codec);dt=time.monotonic()-t
   try:
    if decompress(payload,codec)!=raw:raise ValueError("codec roundtrip mismatch")
   except Exception as ex:
    options.append(dict(kind=kind,codec=codec,valid=False,error=repr(ex)));continue
   blob=pack(md,payload,kind,codec)
   options.append(dict(kind=kind,codec=codec,bytes=len(blob),encode_seconds=dt))
   if best is None or len(blob)<len(best):best=blob
 return best,dict(encode_seconds=time.monotonic()-start,options=options)
def decode(blob,tokenizer=None,temporary_root=None,qwen_tokenizer=None):
 assert blob[:8]==MAGIC
 hlen=struct.unpack_from("<I",blob,8)[0];md=json.loads(decompress(blob[12:12+hlen],"zstd19"));payload=blob[12+hlen:];kind=md["kind"]
 if kind in ["scoped","global","shuffled"]:
  n=struct.unpack_from("<I",payload)[0];route=decompress(payload[4:4+n],"zstd19")
  with tempfile.TemporaryDirectory(prefix="ccm-read-",dir=temporary_root) as d:
   d=Path(d);(d/"coded").write_bytes(route+payload[4+n:])
   subprocess.run([str(BINARY),"decode",str(d/"coded"),str(d/"decoded"),str(md["bits"]),str(int(kind!="global"))],check=True)
   raw=(d/"decoded").read_bytes()
  count=struct.unpack_from("<I",raw)[0];pos=4+8*count;bodies=[]
  for i in range(count):
   _,length=struct.unpack_from("<II",raw,4+8*i);bodies.append(raw[pos:pos+length].decode());pos+=length
  assert pos==len(raw)
 else:
  raw=decompress(payload,md["codec"]);count=struct.unpack_from("<I",raw)[0];bodies=[]
  if kind=="utf8":
   pos=4+4*count
   for i in range(count):
    n=struct.unpack_from("<I",raw,4+4*i)[0];bodies.append(raw[pos:pos+n].decode());pos+=n
  else:
   pos=4
   for i in range(count):
    n,ne=struct.unpack_from("<II",raw,pos);pos+=8;size=4 if kind=="qwen_tokens" else 2
    ids=np.frombuffer(raw[pos:pos+size*n],"<u4" if size==4 else "<u2").tolist();pos+=size*n
    active_tok=qwen_tokenizer if kind=="qwen_tokens" else tokenizer
    text=active_tok.decode(ids,skip_special_tokens=False)
    if ne:
     z=zlib.decompressobj(zdict=text.encode()[-32768:]);text=(z.decompress(raw[pos:pos+ne])+z.flush()).decode()
    pos+=ne;bodies.append(text)
  assert pos==len(raw)
 assert count==len(md["metadata"])
 return [dict(meta,body=text) for meta,text in zip(md["metadata"],bodies)]
