"""Lexical cross-tokenizer interfaces extracted from the research implementation."""
import torch
from torch import nn
from torch.nn import functional as F
from tokenizers import Tokenizer,normalizers

class _TransportConnector(nn.Module):

    def __init__(self, source_embeddings, host_embeddings, lexicon):
        super().__init__()
        self.register_buffer('source', F.normalize(source_embeddings.detach().float(), dim=-1))
        self.register_buffer('ids', lexicon['ids'].to(source_embeddings.device))
        self.register_buffer('lengths', lexicon['lengths'].to(source_embeddings.device))
        object.__setattr__(self, 'host_embeddings', host_embeddings)
        self.zero_values = False

def boundary_lexicon(source_tokenizer, host_tokenizer, host):
    from tokenizers import Tokenizer, normalizers
    tokenizer = Tokenizer.from_str(host_tokenizer.backend_tokenizer.to_str())
    if host == 'B':
        tokenizer.normalizer = normalizers.Replace(' ', '▁')
    pieces = source_tokenizer.convert_ids_to_tokens(list(range(len(source_tokenizer))))
    strings = [(' ' if p.startswith('▁') else '') + source_tokenizer.decode([i], skip_special_tokens=True) for (i, p) in enumerate(pieces)]
    encoded = tokenizer.encode_batch(strings, add_special_tokens=False)
    rows = [x.ids or [host_tokenizer.eos_token_id] for x in encoded]
    width = max(map(len, rows))
    ids = torch.full((len(rows), width), host_tokenizer.eos_token_id, dtype=torch.long)
    for (i, row) in enumerate(rows):
        ids[i, :len(row)] = torch.tensor(row)
    return dict(ids=ids, lengths=torch.tensor(list(map(len, rows))), source_vocab=len(rows), clipped_pieces=0, max_pieces=width, original_input_boundary_preserved=True)

class ValidVocabularyTransport(_TransportConnector):
    def forward(self,z):
        sim=F.normalize(z.float(),dim=-1)@self.source.T
        # Model embeddings can contain padding rows beyond the tokenizer vocabulary.
        # Such rows have no token semantics or lexicon entry and must not be candidates.
        sim[:,len(self.lengths):]=-torch.inf
        sim,ix=sim.topk(8,dim=-1)
        w=(sim*100.).softmax(-1)
        lens=self.lengths[ix]
        position=torch.arange(self.ids.shape[1],device=z.device)
        valid=position[None,None,:]<lens[:,:,None]
        mass=(w[:,:,None]*valid).sum(1)
        emb=self.host_embeddings(self.ids[ix]).float()
        value=(emb*w[:,:,None,None]*valid[:,:,:,None]).sum(1)/mass[:,:,None].clamp_min(1e-8)
        out=value[mass>=.5]
        return out*0 if self.zero_values else out


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
  tokens=torch.from_numpy(tokens).to(z.device);valid=torch.from_numpy(valid).to(z.device)
  probability=torch.from_numpy(probability).to(z.device);mass=(probability[:,:,None]*valid).sum(1)
  embeddings=self.host_embeddings(tokens).float()
  value=(embeddings*probability[:,:,None,None]*valid[:,:,:,None]).sum(1)/mass[:,:,None].clamp_min(1e-8)
  out=value[mass>=.5]
  self.last_stats=dict(source_vectors=len(z),word_groups=len(groups),candidate_words=len(strings),beam=4,max_word_entropy=max_entropy,
   full_history_loaded=False,whole_answer_text_path=False)
  return out*0 if self.zero_values else out
