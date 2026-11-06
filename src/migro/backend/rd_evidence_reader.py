"""Query-conditioned evidence selection and conservative null-answer readout.
Uses a public frozen cross encoder; does not learn from benchmark answers.
"""
import re,time,math
from collections import Counter
import numpy as np
import torch
from torch.nn import functional as F
from transformers import AutoTokenizer,AutoModelForSequenceClassification
from nltk.stem import PorterStemmer
from .rd_dialogue import DialogueReader,words,bind_person_references
from .paths import model_path
class EvidenceRanker:
 def __init__(self,path=None,device="cuda"):
  path=path or model_path("RERANKER","ms-marco-MiniLM-L6-v2")
  self.tokenizer=AutoTokenizer.from_pretrained(path,local_files_only=True)
  self.model=AutoModelForSequenceClassification.from_pretrained(path,local_files_only=True).to(device).eval().requires_grad_(False)
  self.device=device;self.tokens=0;self.calls=0;self.seconds=0.;self.parameter_bytes=sum(p.numel()*p.element_size() for p in self.model.parameters())
 @torch.no_grad()
 def score(self,query,passages):
  start=time.monotonic();values=[]
  for st in range(0,len(passages),32):
   texts=passages[st:st+32]
   x=self.tokenizer([query]*len(texts),texts,padding=True,truncation="only_second",max_length=384,return_tensors="pt").to(self.device)
   values.extend(self.model(**x).logits.flatten().float().cpu().tolist());self.tokens+=int(x.attention_mask.sum());self.calls+=1
  self.seconds+=time.monotonic()-start
  return np.asarray(values)
class EvidenceReader(DialogueReader):
 def __init__(self,state,ranker):
  super().__init__(state);self.ranker=ranker;stemmer=PorterStemmer();self.stem=lambda s:[stemmer.stem(w) for w in words(s)]
  self.bodies=[t.split(" said:",1)[-1].split("; Time reference:",1)[0] for t in self.texts]
  self.terms=[set(self.stem(t)) for t in self.bodies]
  self.rank_text=[]
  for i,t in enumerate(self.bodies):
   author=state.meta[i]["speaker"];other=[s for s in self.speakers if s!=author]
   # Resolve speaker-indexed pronouns for scoring only. The delivered content remains exact.
   canonical=bind_person_references(t,author,other[0] if len(other)==1 else None)
   self.rank_text.append(author+"; "+self.state.meta[i]["date"]+": "+canonical)
  self.separator=self.core.codec.tokenizer.encode("\n",add_special_tokens=False)
 def candidates(self,query,stratified=True,limit=64):
  n=len(self.ids);q=F.normalize(self.core.encode_semantic(["Represent this sentence for searching relevant passages: "+query,query])[:,:384],dim=-1)
  dense=(q@self.keys.T).amax(0).cpu().numpy()
  terms=set(self.stem(query))-set(self.stem(" ".join(self.speakers)))
  lex=np.asarray([len(terms&t)/max(1,len(terms)) for t in self.terms])
  score=dense+.18*lex
  selected=[]
  def add(i):
   if int(i) not in selected and len(selected)<limit:selected.append(int(i))
  # A fixed chronological stratum reserve keeps old candidates in the reranking pool.
  # The rule depends on observation order only, never gold evidence or answer locations.
  if stratified:
   for block in np.array_split(np.arange(n),4):
    for i in sorted(block,key=lambda i:-score[i])[:4]:add(i)
  for i in np.argsort(-score,kind="stable")[:32]:add(i)
  for i in np.argsort(-lex,kind="stable")[:16]:
   if lex[i]>0:add(i)
  for i in np.argsort(-dense,kind="stable"):add(i)
  return selected,score
 def select_evidence(self,query,topk=16,budget=2000,mode="support"):
  start=time.monotonic();candidates,prior=self.candidates(query,stratified=mode in ("support","support_local","support_noedges"))
  logits=self.ranker.score(query,[self.rank_text[i] for i in candidates]);scores=dict(zip(candidates,logits))
  order=sorted(candidates,key=lambda i:(-scores[i],i))
  # Keep a bounded, complementary seed set; relevance is scored independently of new observations.
  seeds=[];remaining=order.copy()
  while remaining and len(seeds)<topk:
   if not seeds:i=remaining[0]
   else:
    v=self.keys[remaining]@self.keys[seeds].T
    penalty=v.amax(1).cpu().numpy()
    i=remaining[int(np.argmax([scores[j] for j in remaining]-.35*penalty))]
   seeds.append(i);remaining.remove(i)
  chosen=[];used=0;edges=[]
  def add(i):
   nonlocal used
   if i in chosen or i<0 or i>=len(self.ids):return False
   size=len(self.ids[i])+len(self.separator)
   if used+size>budget:return False
   chosen.append(i);used+=size;return True
  kept=[i for i in seeds if add(i)]
  if mode in ("cross","support_local"):
   # Generic reranker control retains old unconditional local closure.
   for i in kept[:6]:
    for j in (i-1,i+1):
     if 0<=j<len(self.ids) and self.state.meta[j]["session"]==self.state.meta[i]["session"]:add(j)
  else:
   # A source-native edge is a candidate, not an automatic write/read obligation.
   proposed=[]
   for i in ([] if mode=="support_noedges" else kept[:6]):
    ref=self.state.meta[i].get("reference")
    refs=[ref] if ref is not None else []
    if re.search(r"^\s*(yes|yeah|no|thanks|that's|it |they |he |she |that |those |this )",self.bodies[i],re.I):refs.append(i-1)
    for j in refs:
     if j is None or j<0 or j>=len(self.ids) or self.state.meta[j]["session"]!=self.state.meta[i]["session"]:continue
     # All scored candidates count against the fixed 64-pair reranking budget.
     if j in scores and scores[j]>=scores[i]-2.0:proposed.append((i,j))
   for i,j in proposed:
    if add(j):edges.append([i,j])
  chosen.sort();z,text=self.packet(chosen)
  meta=dict(slots=chosen,seeds=kept,source_tokens=used,source_ids=[self.ids[i] for i in chosen],
   packet_source_ids=[self.ids[i]+self.separator for i in chosen],
   dialogue_ids=[self.state.meta[i]["dialogue_id"] for i in chosen],
   rerank_candidates=len(candidates),candidate_slots=candidates,candidate_scores=[float(scores[i]) for i in candidates],
   query_gated_edges=edges,reader=mode,reader_seconds=time.monotonic()-start,
   original_history_loaded=False,state_mutated=False)
  return z,text,meta
 def packet(self,slots,*args,**kwargs):
  # Keep full original utterances and explicit boundaries in the continuous packet.
  ids=[t for i in slots for t in self.ids[i]+self.separator]
  z=self.core.source.embedding(torch.tensor(ids,device=self.core.device)) if ids else None
  return z,"\n".join(self.texts[i] for i in slots)
NULL_RE=re.compile(r"\b(no information(?: is)? available|not (?:explicitly )?(?:mentioned|provided|specified|stated|given|available)|(?:cannot|can't|unable to) (?:determine|infer|tell)|(?:do not|does not|don't|doesn't) (?:know|specify|mention|provide)|unknown)\b",re.I)
def explicit_null(text):
 return bool(NULL_RE.search(text))
def options_from_prompt(text):
 match=re.search(r"\(a\)\s*(.*?)\s*\(b\)\s*(.*?)(?:\.?\s*Select\s*\(a\).*|$)",text,re.S|re.I)
 return {"a":match.group(1).strip(),"b":match.group(2).strip().rstrip(".")} if match else None
def selective_null(query,original,blind,speakers,strict=False):
 if explicit_null(blind):return "No information available","explicit_unknown"
 if not strict:return original,None
 targets=[s for s in speakers if re.search(r"\b"+re.escape(s)+r"\b",query,re.I)]
 if len(targets)==1 and not re.match(r"\s*who\b",query,re.I):
  target=targets[0];others=[s for s in speakers if s!=target]
  # Only an explicit conflicting attribution; no hard filtering of other speakers' evidence.
  if not re.search(r"\b"+re.escape(target)+r"\b",blind,re.I):
   for other in others:
    if re.match(r"\s*"+re.escape(other)+r"\b",blind,re.I) or re.search(r"\b"+re.escape(other)+r"['’]s\b",blind,re.I):
     return "No information available","conflicting_subject"
 return original,None
