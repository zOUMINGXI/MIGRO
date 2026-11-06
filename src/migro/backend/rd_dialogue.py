"""Observation-level dialogue bindings and a portable, causal handoff envelope.
Content remains explicitly charged lossless source-token residuals.
"""
import json,math,re,struct,zlib,copy
from pathlib import Path
from collections import Counter
from datetime import datetime,timedelta
import numpy as np
import torch
from torch.nn import functional as F
from .rd_shared_anchors import SharedAnchorMemory,AnchorConfig
from .rd_benchmark import BenchmarkBridge
from .host_bridge import MARKER
R=Path("research/rd_dialogue_handoff_20261025")
GROUP="rd-dialogue-handoff-20261025"
DEV=("conv-26","conv-30")
STOP=set("a an the is are was were be been being am i me my we our us you your he she his her they their it its that this these those what when where who which how why would could should do does did have has had of in on at to for from and or but if as with about".split())
def words(text):
 return [w for w in re.findall(r"[a-z0-9]+",text.lower()) if w not in STOP]
def observation_stream(conversation):
 out=[]
 sessions=sorted([k for k in conversation if re.fullmatch(r"session_\d+",k)],key=lambda x:int(x.split("_")[1]))
 for session in sessions:
  date=conversation[session+"_date_time"]
  for turn in conversation[session]:
   body=turn["text"]
   if turn.get("blip_caption"):body+=" [Shared image description: "+turn["blip_caption"]+"]"
   out.append(dict(speaker=turn["speaker"],date=date,session=session,dialogue_id=turn["dia_id"],body=body,role="observed_human"))
 return out
def observation_text(o):
 return "DATE: "+o["date"]+"; "+o["speaker"]+" said: "+o["body"]
def temporal_notes(o):
 try:date=datetime.strptime(o["date"].split(" on ")[-1],"%d %B, %Y")
 except ValueError:return ""
 text=o["body"].lower();notes=[]
 for term,days in (("yesterday",1),("the day before yesterday",2),("tomorrow",-1)):
  if re.search(r"\b"+term+r"\b",text) and not(term=="yesterday" and "the day before yesterday" in text):
   notes.append(term+" = "+(date-timedelta(days=days)).strftime("%-d %B %Y"))
 if "last year" in text:notes.append("last year = "+str(date.year-1))
 return "; ".join(notes)
def bind_person_references(text,author,addressee=None):
 # Preserve double-quoted literal spans; leave ambiguous we/our and 'd unresolved.
 pairs=[(r"\bI['’]m\b",author+" is"),(r"\bI am\b",author+" is"),
  (r"\bI['’]ve\b",author+" has"),(r"\bI have\b",author+" has"),
  (r"\bI['’]ll\b",author+" will"),(r"\bmyself\b",author),
  (r"\bmine\b",author+"'s"),(r"\bmy\b",author+"'s"),(r"\bme\b",author),
  (r"\bI\b(?!['’])",author)]
 if addressee:
  pairs += [(r"\byou['’]re\b",addressee+" is"),(r"\byou are\b",addressee+" is"),
   (r"\byou['’]ve\b",addressee+" has"),(r"\byou have\b",addressee+" has"),
   (r"\byou['’]ll\b",addressee+" will"),(r"\byourself\b",addressee),
   (r"\byours\b",addressee+"'s"),(r"\byour\b",addressee+"'s"),(r"\byou\b(?!['’])",addressee)]
 chunks=re.split(r'("[^"]*"|“[^”]*”)',text)
 for j in range(0,len(chunks),2):
  for pattern,repl in pairs:chunks[j]=re.sub(pattern,lambda _:repl,chunks[j],flags=re.I)
 return "".join(chunks)
class DialogueState:
 def __init__(self,core,key_mode="full"):
  self.key_mode=key_mode;self.core=core;self.mem=SharedAnchorMemory(core,AnchorConfig());self.meta=[];self.events=[]
 def append_many(self,obs,writer="canonical",links=None):
  texts=[observation_text(o)+(("; Time reference: "+temporal_notes(o)) if temporal_notes(o) else "") for o in obs]
  keys=self.core.encode_semantic([o["speaker"]+": "+o["body"] for o in obs] if self.key_mode=="content" else texts)
  for j,(o,text,key) in enumerate(zip(obs,texts,keys)):
   ids=self.core.codec.tokenizer.encode(text,add_special_tokens=False)
   self.mem.write(key,torch.tensor(ids,device=self.core.device),text)
   i=len(self.meta);ref=links[j] if links is not None else None
   # Metadata contains identities/links, never evaluation answers or queries.
   self.meta.append(dict(speaker=o["speaker"],session=o["session"],dialogue_id=o["dialogue_id"],date=o["date"],
    role=o["role"],writer=writer,reference=ref))
 def serialize(self):
  header=dict(schema="dialogue-bindings-v1",meta=self.meta,events=self.events)
  if self.key_mode!="full":header["key_mode"]=self.key_mode
  meta=json.dumps(header,separators=(",",":")).encode()
  meta=zlib.compress(meta,6);blob=self.mem.serialize()
  return struct.pack("<Q",len(meta))+meta+blob
 def restore(self,blob):
  n=struct.unpack("<Q",blob[:8])[0];d=json.loads(zlib.decompress(blob[8:8+n]));assert d["schema"]=="dialogue-bindings-v1"
  self.key_mode=d.get("key_mode","full");self.meta=d["meta"];self.events=d["events"];self.mem.restore(blob[8+n:]);assert self.mem.n==len(self.meta);return self
 def handoff(self,target):
  # The transfer itself does not mutate, convert or regenerate the user memory.
  blob=self.serialize();other=DialogueState(self.core).restore(blob)
  assert other.serialize()==blob
  return other
class DialogueReader:
 def __init__(self,state):
  self.state=state;self.core=state.core;self.mem=state.mem
  self.ids=[self.mem.read_slot(i,False).cpu().tolist() for i in range(self.mem.n)]
  self.texts=[self.core.codec.tokenizer.decode(i,skip_special_tokens=True) for i in self.ids]
  self.tf=[Counter(words(t)) for t in self.texts];self.df=Counter(w for t in self.tf for w in t)
  self.speakers=sorted(set(x["speaker"] for x in state.meta))
  self.guided_ids=[];self.canonical_ids=[]
  for i,ids in enumerate(self.ids):
   author=state.meta[i]["speaker"];other=[x for x in self.speakers if x!=author]
   guide="[Quote author: "+author+". I/my refers to "+author+". You/your refers to "+(other[0] if len(other)==1 else "the addressee")+".] "
   self.guided_ids.append(self.core.codec.tokenizer.encode(guide,add_special_tokens=False)+ids)
   original=self.texts[i];parts=original.split(" said:",1)
   if len(parts)==2:
    normalized=parts[0]+" said: "+bind_person_references(parts[1],author,other[0] if len(other)==1 else None)
   else:normalized=original
   self.canonical_ids.append(self.core.codec.tokenizer.encode(normalized,add_special_tokens=False))
  self.lens=np.array([sum(t.values()) for t in self.tf]);self.avg=float(self.lens.mean())
  # Reconstructed address workspace; no original histories or host KV loaded.
  self.keys=F.normalize(self.mem.coordinates()@self.mem.full_basis.T+self.mem.mean,dim=-1)
 def packet(self,slots,structured=False,canonical_person=False):
  source=self.canonical_ids if canonical_person else self.guided_ids if structured else self.ids
  ids=[t for i in slots for t in source[i]]
  z=self.core.source.embedding(torch.tensor(ids,device=self.core.device)) if ids else None
  return z,"\n".join(self.core.codec.tokenizer.decode(source[i],skip_special_tokens=True) for i in slots)
 def select(self,query,topk=10,budget=1300,closure=True,hybrid=True,native_edges=True,diverse=False,owner_scope=False,canonical_person=False):
  q=F.normalize(self.core.encode_semantic(["Represent this sentence for searching relevant passages: "+query,query])[:,:384],dim=-1)
  dense=(q@self.keys.T).amax(0).cpu().numpy()
  qs=words(query);n=len(self.tf)
  targets=[name for name in self.speakers if re.search(r"\b"+re.escape(name)+r"\b",query,re.I)]
  owner=targets[0] if owner_scope and len(targets)==1 else None
  allowed=np.array([owner is None or x["speaker"]==owner for x in self.state.meta])
  packet_ids=self.canonical_ids if canonical_person else self.guided_ids if owner_scope else self.ids
  bm=np.zeros(n)
  for w in set(qs):
   df=self.df.get(w,0)
   if not df:continue
   f=np.array([t.get(w,0) for t in self.tf])
   bm+=math.log(1+(n-df+.5)/(df+.5))*f*2.2/(f+1.2*(.25+.75*self.lens/max(1,self.avg)))
  ranks=np.empty(n);ranks[np.argsort(-dense,kind="stable")]=np.arange(n)
  scores=1/(30+ranks)
  if hybrid and bm.max()>0:
   lr=np.empty(n);lr[np.argsort(-bm,kind="stable")]=np.arange(n);scores+=.65/(30+lr)*(bm>0)
  scores[~allowed]=-np.inf
  order=[i for i in np.argsort(-scores,kind="stable").tolist() if allowed[i]]
  if diverse:
   # Select semantically relevant but complementary evidence, not ten paraphrases.
   relevance=dense+.12*(bm/max(1e-8,bm.max()))
   relevance[~allowed]=-np.inf
   candidates=[i for i in np.argsort(-relevance,kind="stable").tolist() if allowed[i]][:100]
   ordered=[];key=self.keys[candidates];sim=(key@key.T).cpu().numpy();penalty=np.zeros(len(candidates))
   while len(ordered)<min(topk,len(candidates)):
    merit=np.array([relevance[i] for i in candidates])-.18*penalty
    for j in ordered:merit[j]=-np.inf
    j=int(merit.argmax());ordered.append(j);penalty=np.maximum(penalty,sim[:,j])
   order=[candidates[j] for j in ordered]+[i for i in order if i not in {candidates[j] for j in ordered}]
  chosen=[];used=0;seeds=[]
  def add(i):
   nonlocal used
   if i in chosen or i<0 or i>=n:return False
   if used+len(packet_ids[i])>budget:return False
   chosen.append(i);used+=len(packet_ids[i]);return True
  for i in order:
   if len(seeds)>=topk:break
   if add(i):seeds.append(i)
  if closure:
   for i in seeds[:6]:
    ref=self.state.meta[i].get("reference")
    if native_edges and ref is not None:add(ref)
    for j in (i-1,i+1):
     if 0<=j<n and self.state.meta[j]["session"]==self.state.meta[i]["session"]:add(j)
  chosen.sort()
  z,text=self.packet(chosen,owner_scope,canonical_person)
  return z,text,dict(slots=chosen,seeds=seeds,source_tokens=used,
   canonical_person=canonical_person,owner_scope=owner_scope,query_owner=owner,structured_speaker_guides=owner_scope,packet_source_ids=[packet_ids[i] for i in chosen],
   dialogue_ids=[self.state.meta[i]["dialogue_id"] for i in chosen],source_ids=[self.ids[i] for i in chosen],
   address_workspace_bytes=self.keys.numel()*self.keys.element_size(),original_history_loaded=False)
READER_RULES=("Answer concisely with only the answer, not a restatement or supporting quotation. "
 "Connect relevant statements by speaker and date. Resolve relative dates against the statement's DATE. "
 "For a question explicitly asking what is likely or would happen, make the supported inference from the person's stated interests and circumstances. "
 "For a factual question, do not invent missing events or details.")
class DialogueBridge(BenchmarkBridge):
 def __init__(self,path,host,policy="bound"):
  super().__init__(path,host);self.policy=policy
 def render(self,query,content=None,soft=False):
  row=copy.deepcopy(query)
  if self.policy=="concise":
   row["system"]=(row.get("system") or "")+"\n"+READER_RULES+" Output only the minimal answer phrase or comma-separated list. Never add an explanation, date citation, or quotation. If choices are provided, return the letter of the best supported choice; choose the no-information option when neither the stated fact nor a justified inference is supported."
  if self.policy in ("bound","concise"):
   row["messages"][-1]["content"]+="\n\n"+READER_RULES
  return super().render(row,content,soft)
