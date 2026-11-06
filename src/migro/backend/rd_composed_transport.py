"""Boundary-composed soft vocabulary transport.
An engineering repair for cross-tokenizer subword fragmentation. Explicit source
residuals remain lexical; this is not a claim of nonlexical semantic compression.
"""
import torch
from torch.nn import functional as F
from tokenizers import Tokenizer,normalizers
from .rd_transport import ValidVocabularyTransport
class ComposedTransport(ValidVocabularyTransport):
 def __init__(self,source_embeddings,host_embeddings,lexicon,source_tokenizer,host_tokenizer,host):
  super().__init__(source_embeddings,host_embeddings,lexicon)
  pieces=source_tokenizer.convert_ids_to_tokens(list(range(len(source_tokenizer))))
  self.pieces=[(" " if p.startswith("▁") else "")+source_tokenizer.decode([i],skip_special_tokens=True) for i,p in enumerate(pieces)]
  self.boundaries=torch.tensor([p.startswith("▁") for p in pieces],device=source_embeddings.device)
  self.backend=Tokenizer.from_str(host_tokenizer.backend_tokenizer.to_str())
  if host=="B":self.backend.normalizer=normalizers.Replace(" ","▁")
  self.host_name=host;self.eos_id=host_tokenizer.eos_token_id
  self.last_stats={};self.cache_enabled=False;self._query_cache={}
 def clear_query_cache(self):
  self._query_cache.clear()
 @torch.no_grad()
 def forward(self,z):
  key=(id(z),self.zero_values)
  if self.cache_enabled and key in self._query_cache:
   old,value=self._query_cache[key]
   if old is z:return value
  value=self._compose(z)
  if self.cache_enabled:
   if len(self._query_cache)>=32:self._query_cache.pop(next(iter(self._query_cache)))
   self._query_cache[key]=(z,value)
  return value
 @torch.no_grad()
 def _compose(self,z):
  sim=F.normalize(z.float(),dim=-1)@self.source.T;sim[:,len(self.lengths):]=-torch.inf
  sim,ix=sim.topk(8,dim=-1);weight=(100*sim).softmax(-1)
  boundaries=(weight*self.boundaries[ix]).sum(-1)>=.5
  ids=ix.cpu().tolist();ws=weight.cpu().tolist();bs=boundaries.cpu().tolist()
  groups=[];beam=[("",1.)]
  for choices,probs,newword in zip(ids,ws,bs):
   if newword and any(s.strip() for s,_ in beam):
    groups.append(beam);beam=[("",1.)]
   expanded={}
   for partial,p in beam:
    for i,w in zip(choices,probs):
     if w<1e-8:continue
     s=partial+self.pieces[i];expanded[s]=expanded.get(s,0.)+p*w
   beam=sorted(expanded.items(),key=lambda x:-x[1])[:4]
   mass=sum(w for _,w in beam)
   beam=[(s,w/max(mass,1e-30)) for s,w in beam]
  if any(s.strip() for s,_ in beam):groups.append(beam)
  if not groups:return torch.zeros(0,self.host_embeddings.weight.shape[1],device=z.device)
  strings=[s for g in groups for s,_ in g]
  encoded=self.backend.encode_batch(strings,add_special_tokens=False)
  # Build ragged word candidates on CPU, then perform one batched GPU mixture.
  import numpy as np,math
  rows=[e.ids for e in encoded];width=max(map(len,rows),default=0)
  if not width:return torch.zeros(0,self.host_embeddings.weight.shape[1],device=z.device)
  count=len(groups);k=max(map(len,groups))
  tokens=np.full((count,k,width),self.eos_id,dtype=np.int64)
  valid=np.zeros((count,k,width),dtype=np.float32);probability=np.zeros((count,k),dtype=np.float32)
  cursor=0;max_entropy=0.
  for g,group in enumerate(groups):
   max_entropy=max(max_entropy,-sum(w*math.log(max(w,1e-30)) for _,w in group))
   for j,(_,w) in enumerate(group):
    row=rows[cursor];cursor+=1;tokens[g,j,:len(row)]=row;valid[g,j,:len(row)]=1;probability[g,j]=w
  # Word groups are independent here; preserve each group's reduction exactly.
  # Bound the temporary [group,beam,width,hidden] tensor for long read budgets.
  blocks=[]
  for start in range(0,count,128):
   tt=torch.from_numpy(tokens[start:start+128]).to(z.device)
   vv=torch.from_numpy(valid[start:start+128]).to(z.device)
   pp=torch.from_numpy(probability[start:start+128]).to(z.device)
   mass=(pp[:,:,None]*vv).sum(1)
   embeddings=self.host_embeddings(tt).float()
   value=(embeddings*pp[:,:,None,None]*vv[:,:,:,None]).sum(1)/mass[:,:,None].clamp_min(1e-8)
   blocks.append(value[mass>=.5])
  out=torch.cat(blocks,dim=0)
  self.last_stats=dict(source_vectors=len(z),word_groups=len(groups),candidate_words=len(strings),beam=4,max_word_entropy=max_entropy,
   full_history_loaded=False,whole_answer_text_path=False)
  return out*0 if self.zero_values else out
