"""Frozen-host input connector. No writer or history replay in this module."""
import math,time,os
import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoTokenizer,AutoModelForCausalLM
SYSTEM="Answer the question using the available memory. Return only a brief answer, with no explanation. If the information is absent or revoked, return unknown."
MARKER="__PORTABLE_MEMORY_INSERTION__"
class Connector(nn.Module):
 def __init__(self,width,embedding_rms,hidden=512,expansion=1):
  super().__init__()
  self.width=width;self.expansion=expansion
  self.norm=nn.LayerNorm(512);self.up=nn.Linear(512,hidden);self.down=nn.Linear(hidden,width*expansion)
  self.log_scale=nn.Parameter(torch.tensor(math.log(embedding_rms)))
 def forward(self,z):
  x=self.down(F.gelu(self.up(self.norm(z.float())))).reshape(-1,self.width)
  return F.normalize(x,dim=-1)*math.sqrt(x.shape[-1])*self.log_scale.exp()
class HostBridge:
 def __init__(self,path,hidden=512):
  self.tokenizer=AutoTokenizer.from_pretrained(path,local_files_only=True)
  self.host=AutoModelForCausalLM.from_pretrained(path,local_files_only=True,dtype=torch.bfloat16,attn_implementation="sdpa").to(os.environ.get("MIGRO_DEVICE", "cuda")).eval().requires_grad_(False)
  self.device=self.host.device
  with torch.no_grad():rms=float(self.host.get_input_embeddings().weight.float().square().mean().sqrt())
  self.connector=Connector(self.host.config.hidden_size,rms,hidden).to(self.device)
  self.host_path=path;self.bind_packet=False
  ids=self.host.generation_config.eos_token_id
  self.eos=set(ids if isinstance(ids,list) else [ids])
  if self.tokenizer.eos_token_id is not None:self.eos.add(self.tokenizer.eos_token_id)
 def synchronize(self):
  if self.device.type == "cuda":
   torch.cuda.synchronize(self.device)
 def ids(self,text):return self.tokenizer.encode(text,add_special_tokens=False)
 def render(self,query,content=None,soft=False):
  soft_content=("Saved question and answer:\nQuestion: "+query+"\nAnswer: "+MARKER) if self.bind_packet else MARKER
  msg=(("<parametric_memory>\n"+soft_content+"\n</parametric_memory>\n\n") if soft else ("<parametric_memory>\n"+content+"\n</parametric_memory>\n\n") if content is not None else "")+"Question: "+query
  return self.tokenizer.apply_chat_template([dict(role="system",content=SYSTEM),dict(role="user",content=msg)],tokenize=False,add_generation_prompt=True)
 def sequence(self,query,z=None,text=None,answer=None):
  emb=self.host.get_input_embeddings();soft_n=0
  if z is not None and len(z):
   before,after=self.render(query,soft=True).split(MARKER)
   left=torch.tensor(self.ids(before),device=self.device);right=torch.tensor(self.ids(after),device=self.device)
   values=self.connector(z.to(self.device)).to(emb.weight.dtype);soft_n=len(values)
   x=torch.cat([emb(left),values,emb(right)],0);n=len(left)+len(right)
  else:
   ids=torch.tensor(self.ids(self.render(query,content=text)),device=self.device)
   x=emb(ids);n=len(ids)
  labels=torch.full((len(x),),-100,device=self.device,dtype=torch.long)
  ans_n=0
  if answer is not None:
   target=self.ids(answer)+[self.tokenizer.eos_token_id];ans_n=len(target)
   a=torch.tensor(target,device=self.device);x=torch.cat([x,emb(a)],0);labels=torch.cat([labels,a])
  return x,labels,dict(prompt_tokens=n,soft_tokens=soft_n,answer_tokens=ans_n,sequence_tokens=len(x))
 def pack(self,items):
  # Left padding: final prompt position is real for every member, including empty.
  n=max(len(x[0]) for x in items)
  embeds=torch.stack([F.pad(x,(0,0,n-len(x),0)) for x,_,_ in items])
  mask=torch.stack([torch.arange(n,device=self.device)>=n-len(x) for x,_,_ in items]).long()
  labels=torch.stack([F.pad(y,(n-len(y),0),value=-100) for _,y,_ in items])
  positions=(mask.cumsum(-1)-1).clamp_min(0)
  return dict(inputs_embeds=embeds,attention_mask=mask,position_ids=positions),labels
 def loss(self,rows,latents):
  items=[self.sequence(r["query"],z=z,answer=r["answer"]) for r,z in zip(rows,latents)]
  inputs,labels=self.pack(items)
  result=self.host(**inputs,labels=labels,use_cache=False)
  return result.loss,[x[2] for x in items]
 @torch.no_grad()
 def generate(self,rows,latents=None,texts=None,max_new=48):
  self.synchronize();e2e_start=time.monotonic()
  if latents is None:latents=[None]*len(rows)
  if texts is None:texts=[None]*len(rows)
  items=[self.sequence(r["query"],z=z,text=t) for r,z,t in zip(rows,latents,texts)]
  inputs,_=self.pack(items);mask=inputs["attention_mask"];positions=inputs["position_ids"]
  self.synchronize();start=time.monotonic()
  interface_seconds=start-e2e_start
  out=self.host(**inputs,use_cache=True,**({"logits_to_keep":1} if getattr(self,"last_logits_only",False) else {}));self.synchronize();prefill=time.monotonic()-start
  cache=out.past_key_values;outputs=[[] for _ in rows];done=torch.zeros(len(rows),dtype=torch.bool,device=self.device)
  for step in range(max_new):
   token=out.logits[:,-1].argmax(-1)
   for i,t in enumerate(token.tolist()):
    if not done[i] and t not in self.eos:outputs[i].append(t)
   done|=torch.tensor([int(t) in self.eos for t in token.tolist()],device=self.device)
   if done.all() or step+1==max_new:break
   mask=torch.cat([mask,torch.ones((len(rows),1),dtype=mask.dtype,device=self.device)],1)
   pos=mask.sum(-1,keepdim=True)-1
   out=self.host(input_ids=token[:,None],attention_mask=mask,position_ids=pos,past_key_values=cache,use_cache=True)
   cache=out.past_key_values
  self.synchronize()
  costs=dict(seconds=time.monotonic()-start,end_to_end_seconds=time.monotonic()-e2e_start,interface_and_pack_seconds=interface_seconds,prefill_seconds=prefill,generated_tokens=sum(map(len,outputs)),
   prompt_tokens=sum(x[2]["prompt_tokens"] for x in items),soft_tokens=sum(x[2]["soft_tokens"] for x in items),
   padded_prefill_tokens=len(rows)*inputs["inputs_embeds"].shape[1],decode_forwards=step,prefill_forwards=1)
  return [self.tokenizer.decode(o,skip_special_tokens=True).strip() for o in outputs],costs
