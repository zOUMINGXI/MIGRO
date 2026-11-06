"""Frozen batched causal content coding. No teacher, training or retained attention KV.
The bitstream is implementation-specific: model/dtype/attention backend/batch are part
of the decoding contract. Real independent decoding is mandatory before use.
"""
import json,struct,time,copy,math
from pathlib import Path
import numpy as np
import torch
from .causal_content_codec import compress,decompress,metadata
TOTAL=1<<24
MAGIC=b"NCV12802"
from .paths import model_path
MODEL=Path(model_path("QWEN","Qwen3-4B-Instruct-2507"))
class Encoder:
 def __init__(self):self.low=0;self.high=(1<<32)-1;self.pending=0;self.bits=[]
 def bit(self,x):self.bits.append(x);self.bits.extend([1-x]*self.pending);self.pending=0
 def put(self,lo,hi):
  w=self.high-self.low+1;self.high=self.low+w*hi//TOTAL-1;self.low+=w*lo//TOTAL
  while True:
   if self.high<(1<<31):self.bit(0)
   elif self.low>=1<<31:self.bit(1);self.low-=1<<31;self.high-=1<<31
   elif self.low>=1<<30 and self.high<3<<30:self.pending+=1;self.low-=1<<30;self.high-=1<<30
   else:break
   self.low*=2;self.high=self.high*2+1
 def finish(self):
  x=copy.copy(self);x.bits=self.bits.copy();x.pending+=1;x.bit(0 if x.low<1<<30 else 1)
  x.bits.extend([0]*(-len(x.bits)%8))
  return np.packbits(np.asarray(x.bits,np.uint8)).tobytes()
class Decoder:
 def __init__(self,blob):
  self.blob=blob;self.pos=0;self.low=0;self.high=(1<<32)-1;self.value=0
  for _ in range(32):self.value=self.value*2+self.bit()
 def bit(self):
  v=(self.blob[self.pos//8]>>(7-self.pos%8))&1 if self.pos//8<len(self.blob) else 0
  self.pos+=1;return v
 def count(self):return ((self.value-self.low+1)*TOTAL-1)//(self.high-self.low+1)
 def consume(self,lo,hi):
  w=self.high-self.low+1;self.high=self.low+w*hi//TOTAL-1;self.low+=w*lo//TOTAL
  while True:
   if self.high<1<<31:pass
   elif self.low>=1<<31:self.low-=1<<31;self.high-=1<<31;self.value-=1<<31
   elif self.low>=1<<30 and self.high<3<<30:self.low-=1<<30;self.high-=1<<30;self.value-=1<<30
   else:break
   self.low*=2;self.high=self.high*2+1;self.value=self.value*2+self.bit()
def cdf(logits):
 p=logits.float().softmax(-1);v=p.shape[-1]
 f=torch.floor(p*(TOTAL-v)).to(torch.int32)+1
 f.scatter_add_(-1,p.argmax(-1,keepdim=True),(TOTAL-f.sum(-1,dtype=torch.int32))[:,None])
 return torch.cumsum(f,-1,dtype=torch.int32)
def route(meta,lengths,page_size,mode="causal"):
 """Chronological page creation; scope grouping only uses already assigned row scopes."""
 pages=[];active={}
 for ri,(row,n) in enumerate(zip(meta,lengths)):
  scope=row.get("scope",row.get("session",""));offset=0
  if mode=="reset":active.pop(scope,None)
  while offset<n+1:
   if scope not in active or len(pages[active[scope]])==page_size:
    active[scope]=len(pages);pages.append([])
   p=pages[active[scope]];count=min(page_size-len(p),n+1-offset)
   p.extend((ri,i,i==n) for i in range(offset,offset+count));offset+=count
 return pages
def make_header(rows,lengths,mode,page_size,batch,escapes):
 return dict(version=1,model="Qwen3-4B-Instruct-2507",dtype="bfloat16",attention="sdpa",batch=batch,
   page_size=page_size,cdf_bits=24,mode=mode,schedule="length_descending",metadata=metadata(rows),lengths=lengths,escapes=escapes,
   context="Earlier content from the same scope, up to 1024 native tokens per independent page; no future text",
   external_predictor_state_bytes=0)
def pack(header,payloads):
 h=compress(json.dumps(header,ensure_ascii=False,separators=(",",":")).encode(),"zstd19")
 sizes=compress(np.asarray([len(x) for x in payloads],"<u4").tobytes(),"zstd19")
 return MAGIC+struct.pack("<III",len(h),len(payloads),len(sizes))+h+sizes+b"".join(payloads)
def unpack(blob):
 assert blob[:8]==MAGIC
 nh,npages,ns=struct.unpack_from("<III",blob,8);header=json.loads(decompress(blob[20:20+nh],"zstd19"))
 sizes=np.frombuffer(decompress(blob[20+nh:20+nh+ns],"zstd19"),dtype="<u4");assert len(sizes)==npages
 pos=20+nh+ns;payloads=[]
 for n in sizes:payloads.append(blob[pos:pos+int(n)]);pos+=int(n)
 assert pos==len(blob)
 return header,payloads
class NeuralContent:
 def __init__(self,batch=64,page_size=1024):
  from transformers import AutoTokenizer,AutoModelForCausalLM
  torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
  self.batch=batch;self.page_size=page_size
  t=time.monotonic();self.tokenizer=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
  self.model=AutoModelForCausalLM.from_pretrained(MODEL,local_files_only=True,dtype=torch.bfloat16,attn_implementation="sdpa").cuda().eval().requires_grad_(False)
  self.bos=self.tokenizer.eos_token_id;self.model_load_seconds=time.monotonic()-t
 def tokenize(self,rows):
  tokens=self.tokenizer([r["body"] for r in rows],add_special_tokens=False)["input_ids"];escapes={}
  for i,(ids,r) in enumerate(zip(tokens,rows)):
   if self.tokenizer.decode(ids,skip_special_tokens=False)!=r["body"]:escapes[str(i)]=r["body"]
  return tokens,escapes
 def encode_prefixes(self,rows,prefixes,mode="causal",progress=None):
  t=time.monotonic();tokens,escapes=self.tokenize(rows);lengths=list(map(len,tokens));meta=metadata(rows)
  pages=route(meta,lengths,self.page_size,mode);cuts={}
  for n in prefixes:
   pp=route(meta[:n],lengths[:n],self.page_size,mode)
   for pi,p in enumerate(pp):cuts.setdefault(pi,set()).add(len(p))
  streams={};stream_bits={};coded_tokens=0
  order=sorted(range(len(pages)),key=lambda i:(-len(pages[i]),i))
  for base in range(0,len(pages),self.batch):
   indices=order[base:base+self.batch];batch=[pages[i] for i in indices];coders=[Encoder() for _ in batch];bits=[0. for _ in batch]
   cache=None;previous=torch.full((self.batch,1),self.bos,device="cuda",dtype=torch.long)
   starts=np.zeros(self.batch,dtype=np.int64);maxlen=max(map(len,batch))
   for j in range(maxlen):
    target=[];known=[];active=[]
    for k in range(self.batch):
     valid=k<len(batch) and j<len(batch[k]);active.append(valid)
     if valid:
      ri,off,is_sep=batch[k][j];target.append(self.bos if is_sep else tokens[ri][off]);known.append(is_sep)
      if off==0:starts[k]=j
     else:target.append(self.bos);known.append(True)
    kwargs={}
    if mode=="masked_reset_legacy":
     allowed=torch.arange(j+1,device="cuda")[None,:]>=torch.as_tensor(starts,device="cuda")[:,None]
     kwargs["attention_mask"]=torch.where(allowed,0.,torch.finfo(torch.bfloat16).min).to(torch.bfloat16)[:,None,None,:]
    with torch.inference_mode():
     out=self.model(input_ids=previous,past_key_values=cache,use_cache=True,**kwargs);cache=out.past_key_values
     f=cdf(out.logits[:,0]);y=torch.tensor(target,device="cuda")
     hi=f.gather(1,y[:,None]).squeeze(1);lo=torch.where(y==0,0,f.gather(1,(y-1).clamp_min(0)[:,None]).squeeze(1))
     intervals=torch.stack([lo,hi],1).cpu().numpy()
    for k in range(len(batch)):
     if active[k] and not known[k]:
      lo,hi=map(int,intervals[k]);coders[k].put(lo,hi);coded_tokens+=1;bits[k]-=math.log2((hi-lo)/TOTAL)
     if active[k] and j+1 in cuts[indices[k]]:
      streams[(indices[k],j+1)]=coders[k].finish();stream_bits[(indices[k],j+1)]=bits[k]
    previous=y[:,None]
   if progress:progress(dict(page_group=base//self.batch+1,pages=min(base+self.batch,len(pages)),total_pages=len(pages),native_tokens=coded_tokens,elapsed_seconds=time.monotonic()-t))
  archives={};prefix_bits={}
  for n in prefixes:
   pp=route(meta[:n],lengths[:n],self.page_size,mode)
   h=make_header(rows[:n],lengths[:n],mode,self.page_size,self.batch,{k:v for k,v in escapes.items() if int(k)<n})
   archives[n]=pack(h,[streams[(pi,len(p))] for pi,p in enumerate(pp)])
   prefix_bits[n]=sum(stream_bits[(pi,len(p))] for pi,p in enumerate(pp))
  return archives,dict(encode_seconds=time.monotonic()-t,native_tokens=coded_tokens,pages=len(pages),escape_rows=len(escapes),prefix_finite_cdf_bits=prefix_bits,
   public_model_bytes=sum(p.stat().st_size for p in MODEL.glob("*.safetensors")),peak_gpu_bytes=torch.cuda.max_memory_allocated(),
   attention_KV_persistent_bytes=0,attention_KV_policy="Temporary coding workspace only; released per page group")
 def decode(self,blob,progress=None,prefix_archives=None):
  t=time.monotonic();h,payloads=unpack(blob)
  assert (h["batch"],h["page_size"])==(self.batch,self.page_size)
  pages=route(h["metadata"],h["lengths"],self.page_size,h["mode"]);assert len(pages)==len(payloads)
  prefix_specs=[]
  for size,prefix_blob in (prefix_archives or {}).items():
   ph,pl=unpack(prefix_blob);pp=route(ph["metadata"],ph["lengths"],self.page_size,ph["mode"])
   assert ph["metadata"]==h["metadata"][:size] and ph["lengths"]==h["lengths"][:size]
   assert all(p==pages[i][:len(p)] for i,p in enumerate(pp))
   prefix_specs.append((size,pp,pl))
  tokens=[[None]*n for n in h["lengths"]];native=0
  order=sorted(range(len(pages)),key=lambda i:(-len(pages[i]),i)) if h.get("schedule")=="length_descending" else list(range(len(pages)))
  for base in range(0,len(pages),self.batch):
   indices=order[base:base+self.batch];batch=[pages[i] for i in indices];ds=[Decoder(payloads[i]) for i in indices]
   extra=[[(len(pp[pi]),Decoder(pl[pi])) for _,pp,pl in prefix_specs if pi<len(pp)] for pi in indices]
   cache=None;previous=torch.full((self.batch,1),self.bos,device="cuda",dtype=torch.long);starts=np.zeros(self.batch,dtype=np.int64)
   for j in range(max(map(len,batch))):
    active=[];known=[];counts=[]
    for k in range(self.batch):
     valid=k<len(batch) and j<len(batch[k]);active.append(valid)
     if valid:
      ri,off,sep=batch[k][j];known.append(sep)
      if off==0:starts[k]=j
     else:known.append(True)
     counts.append(ds[k].count() if valid and not known[k] else 0)
    kwargs={}
    if h["mode"]=="masked_reset_legacy":
     allowed=torch.arange(j+1,device="cuda")[None,:]>=torch.as_tensor(starts,device="cuda")[:,None]
     kwargs["attention_mask"]=torch.where(allowed,0.,torch.finfo(torch.bfloat16).min).to(torch.bfloat16)[:,None,None,:]
    with torch.inference_mode():
     out=self.model(input_ids=previous,past_key_values=cache,use_cache=True,**kwargs);cache=out.past_key_values;f=cdf(out.logits[:,0])
     c=torch.tensor(counts,device="cuda",dtype=torch.int32)[:,None]
     y=torch.searchsorted(f,c,right=True).squeeze(1).long()
     hi=f.gather(1,y[:,None]).squeeze(1);lo=torch.where(y==0,0,f.gather(1,(y-1).clamp_min(0)[:,None]).squeeze(1))
     values=torch.stack([y,lo,hi],1).cpu().numpy()
    previous=[]
    for k in range(self.batch):
     if active[k] and not known[k]:
      symbol,lo,hi=map(int,values[k]);ds[k].consume(lo,hi)
      for end,pd in extra[k]:
       if j<end:
        assert lo<=pd.count()<hi,("prefix arithmetic decoder mismatch",base+k,j)
        pd.consume(lo,hi)
      ri,off,_=batch[k][j];tokens[ri][off]=symbol;native+=1
     else:symbol=self.bos
     previous.append(symbol)
    previous=torch.tensor(previous,device="cuda")[:,None]
   if progress:progress(dict(page_group=base//self.batch+1,pages=min(base+self.batch,len(pages)),total_pages=len(pages),native_tokens=native,elapsed_seconds=time.monotonic()-t))
  rows=[dict(r,body=h["escapes"].get(str(i),self.tokenizer.decode(tokens[i],skip_special_tokens=False))) for i,r in enumerate(h["metadata"])]
  return rows,dict(decode_seconds=time.monotonic()-t,native_tokens=native,pages=len(pages),external_history_access=False,verified_prefix_streams=len(prefix_specs),prefix_method="Independent arithmetic decoder states checked against identical decoded-history CDF intervals")
