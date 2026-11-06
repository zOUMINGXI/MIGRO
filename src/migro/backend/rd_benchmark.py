"""Frozen R2 benchmark adapter; the shared state is host-independent."""
import copy,json,sys,time
from pathlib import Path
from types import SimpleNamespace
import torch
from torch import nn
from transformers import AutoConfig,AutoModel,AutoTokenizer
from .rd_compact_core import CompactReadCore,EmbeddingOnly
from .host_bridge import HostBridge,MARKER
R=Path("research/rd_benchmarks_transfer_20261025")
GROUP="rd-benchmarks-transfer-20261025"
from .paths import asset_path,model_path
HOSTS={"A":model_path("QWEN","Qwen3-4B-Instruct-2507"),"B":model_path("PHI","Phi-3-mini-4k-instruct")}
ARMS=("state_only","retrieved_text","off","history","history_memory")
CAPS={"mc":2048,"locomo":50}
def shared_eval():
 import os
 sys.path.insert(0,os.environ["MIGRO_LEGACY_EVALUATOR"])
 import evaluate_benchmarks
 return evaluate_benchmarks
def load_core(device="cuda"):
 root=asset_path("research/rd_shared_pages_20261025/deployment")
 d=torch.load(root/"minimal_core.pt",map_location="cpu",weights_only=True)
 m=CompactReadCore.__new__(CompactReadCore);nn.Module.__init__(m)
 m.semantic_encoder=AutoModel.from_config(AutoConfig.from_pretrained(root/"semantic_config",local_files_only=True))
 m.source=EmbeddingOnly(nn.Embedding.from_pretrained(d["source.embedding.weight"],freeze=True))
 m.semantic_tokenizer=AutoTokenizer.from_pretrained(root/"semantic_tokenizer",local_files_only=True)
 m.codec=SimpleNamespace(model=m.source,tokenizer=AutoTokenizer.from_pretrained(root/"source_tokenizer",local_files_only=True))
 for name in ("native_input_rms","value_basis"):m.register_buffer(name,d[name])
 m.load_state_dict(d);m.device=torch.device(device);m.to(m.device).eval().requires_grad_(False)
 return m

class BenchmarkBridge(HostBridge):
 def __init__(self,path,host):
  super().__init__(path);self.host_name=host
  self.context_limit=min(32768,self.host.config.max_position_embeddings)
  self.audit=[]
 def render_messages(self,msgs):
  return self.tokenizer.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True,enable_thinking=False)
 def render(self,query,content=None,soft=False):
  row=query
  messages=copy.deepcopy(row.get("history",[]))+copy.deepcopy(row["messages"])
  if row.get("system") and not any(x["role"]=="system" for x in messages):
   messages.insert(0,dict(role="system",content=row["system"]))
  if soft or content is not None:
   value=MARKER if soft else content
   prefix="<conversation_memory>\nRecovered earlier conversation fragments, in chronological order:\n"+value+"\n</conversation_memory>\n\n"
   messages[-1]["content"]=prefix+messages[-1]["content"]
  return self.render_messages(messages)
 def build_item(self,row,z=None,text=None):
  emb=self.host.get_input_embeddings()
  if z is not None and len(z):
   before,after=self.render(row,soft=True).split(MARKER)
   left=torch.tensor(self.ids(before),device=self.device)
   right=torch.tensor(self.ids(after),device=self.device)
   values=self.connector(z.to(self.device)).to(emb.weight.dtype)
   x=torch.cat([emb(left),values,emb(right)])
   soft_n=len(values);n=len(left)+len(right)
  else:
   ids=torch.tensor(self.ids(self.render(row,content=text)),device=self.device)
   x=emb(ids);soft_n=0;n=len(ids)
  labels=torch.full((len(x),),-100,device=self.device,dtype=torch.long)
  return x,labels,dict(prompt_tokens=n,soft_tokens=soft_n,answer_tokens=0,sequence_tokens=len(x))
 def sequence(self,query,z=None,text=None,answer=None):
  assert answer is None
  item=self.build_item(query,z,text)
  assert len(item[0])+CAPS[query["benchmark"]]<=self.context_limit,(query["id"],len(item[0]),self.context_limit)
  self.audit.append(item[2])
  return item
 @torch.no_grad()
 def fit_history(self,row,history,z):
  # Identical host-specific history for baseline and augmentation.
  # Reserve the actual continuous packet, with no alteration to current query or packet.
  r={**row,"history":[]}
  base=len(self.build_item(r,z=z)[0]);budget=self.context_limit-CAPS[row["benchmark"]]
  if base>budget:raise ValueError(f"Current query + unchanged packet exceed {row['id']}: {base}>{budget}")
  selected=copy.deepcopy(history)
  total_chars=sum(len(x["content"]) for x in selected)
  while selected:
   r["history"]=selected
   n=len(self.build_item(r,z=z)[0])
   if n<=budget:break
   # Preserve original roles and the most recent contiguous history; explicit oldest-text trimming.
   first=next((i for i,x in enumerate(selected) if x["role"]!="system"),None)
   if first is None:raise ValueError("System-only history exceeds budget")
   ids=self.ids(selected[first]["content"]);drop=n-budget+8
   if len(ids)<=drop:selected.pop(first)
   else:selected[first]["content"]=self.tokenizer.decode(ids[drop:],skip_special_tokens=False)
  r["history"]=selected
  assert len(self.build_item(r,z=z)[0])<=budget
  kept_chars=sum(len(x["content"]) for x in selected)
  return selected,dict(original_history_chars=total_chars,retained_history_chars=kept_chars,
   history_truncated=kept_chars<total_chars,
   history_budget_reserved_for_same_packet=True)

 @torch.no_grad()
 def generate(self,rows,latents=None,texts=None,max_new=48):
  # Native history prefix cache belongs only to the history-access reference protocol.
  # The portable conditions still have no raw-history input or host KV state.
  history=rows[0]["query"].get("history",[])
  if self.host_name!="A" or not history or any(r["query"].get("history")!=history for r in rows):
   return super().generate(rows,latents=latents,texts=texts,max_new=max_new)
  torch.cuda.synchronize();e2e=time.monotonic()
  latents=latents if latents is not None else [None]*len(rows)
  texts=texts if texts is not None else [None]*len(rows)
  items=[self.sequence(r["query"],z=z,text=t) for r,z,t in zip(rows,latents,texts)]
  messages=copy.deepcopy(history);system=rows[0]["query"].get("system")
  if system and not any(m["role"]=="system" for m in messages):messages.insert(0,dict(role="system",content=system))
  prefix=self.tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=False,enable_thinking=False)
  ids=self.ids(prefix);n=len(ids);ix=torch.tensor(ids,device=self.device)
  embeddings=self.host.get_input_embeddings()(ix)
  if not all(len(x)>n and torch.equal(x[:n],embeddings) for x,_,_ in items):
   # No approximate prefix matching.
   self.audit=[]
   return super().generate(rows,latents=latents,texts=texts,max_new=max_new)
  miss=getattr(self,"_history_ids",None)!=ids
  torch.cuda.synchronize();prefill_start=time.monotonic()
  if miss:
   self._history_cache=None
   pref=self.host(input_ids=ix[None],attention_mask=torch.ones(1,n,device=self.device,dtype=torch.long),use_cache=True,logits_to_keep=1)
   self._history_cache=pref.past_key_values;self._history_ids=ids
   del pref
  cache=copy.deepcopy(self._history_cache)
  cache.batch_repeat_interleave(len(rows))
  suffix=[(x[n:],y[n:],meta) for x,y,meta in items]
  inputs,_=self.pack(suffix)
  suffix_mask=inputs["attention_mask"]
  mask=torch.cat([torch.ones(len(rows),n,device=self.device,dtype=torch.long),suffix_mask],1)
  inputs["attention_mask"]=mask
  inputs["position_ids"]=(suffix_mask.cumsum(-1)-1).clamp_min(0)+n
  out=self.host(**inputs,past_key_values=cache,use_cache=True,logits_to_keep=1)
  torch.cuda.synchronize();prefill_seconds=time.monotonic()-prefill_start
  cache=out.past_key_values;outputs=[[] for _ in rows]
  done=torch.zeros(len(rows),device=self.device,dtype=torch.bool)
  for step in range(max_new):
   token=out.logits[:,-1].argmax(-1)
   values=token.tolist()
   for i,t in enumerate(values):
    if not done[i] and t not in self.eos:outputs[i].append(t)
   done|=torch.tensor([t in self.eos for t in values],device=self.device)
   if done.all() or step+1==max_new:break
   mask=torch.cat([mask,torch.ones((len(rows),1),dtype=mask.dtype,device=self.device)],1)
   out=self.host(input_ids=token[:,None],attention_mask=mask,position_ids=mask.sum(-1,keepdim=True)-1,
    past_key_values=cache,use_cache=True,logits_to_keep=1)
   cache=out.past_key_values
  assert self._history_cache.get_seq_length()==n,"Query mutated reusable history prefix"
  torch.cuda.synchronize()
  costs=dict(seconds=time.monotonic()-prefill_start,end_to_end_seconds=time.monotonic()-e2e,
   prefill_seconds=prefill_seconds,interface_and_pack_seconds=prefill_start-e2e,
   prompt_tokens=sum(x[2]["prompt_tokens"] for x in items),soft_tokens=sum(x[2]["soft_tokens"] for x in items),
   generated_tokens=sum(map(len,outputs)),decode_forwards=step,prefill_forwards=1+int(miss),
   padded_prefill_tokens=inputs["inputs_embeds"].shape[1]*len(rows)+n*int(miss),
   cached_native_history_prefix_tokens=n,native_history_cache_reused=not miss,
   reference_prefix_only_no_memory_write=True)
  return [self.tokenizer.decode(o,skip_special_tokens=True).strip() for o in outputs],costs
