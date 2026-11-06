"""Bound-statement scoring: read native reference relations before ranking."""
from collections import defaultdict
import numpy as np
from .rd_evidence_reader import EvidenceReader
class BoundEvidenceReader(EvidenceReader):
 def __init__(self,state,ranker):
  super().__init__(state,ranker)
  standalone=self.rank_text.copy();self.reverse=defaultdict(list)
  for i,m in enumerate(state.meta):
   ref=m.get("reference")
   if ref is not None and 0<=ref<i and state.meta[ref]["session"]==m["session"]:self.reverse[ref].append(i)
   else:ref=None
   other=[s for s in self.speakers if s!=m["speaker"]]
   destination=(" addressed to "+other[0]) if len(other)==1 else ""
   current="Current statement by "+m["speaker"]+destination+": "+standalone[i]
   previous=ref if ref is not None else i-1 if i>0 and state.meta[i-1]["session"]==m["session"] else None
   # Current content is first, so the 384-token ranker limit truncates support first.
   self.rank_text[i]=current+(" Earlier referenced context: "+standalone[previous] if previous is not None else "")
 def candidates(self,query,stratified=True,limit=64):
  base,score=super().candidates(query,stratified=stratified,limit=limit)
  selected=base[:max(1,limit-16)]
  targets=[s for s in self.speakers if s.lower() in query.lower()]
  proposed=[]
  for i in sorted(base,key=lambda i:-score[i])[:16]:
   ref=self.state.meta[i].get("reference")
   if ref is not None:proposed.append(ref)
   children=sorted(self.reverse.get(i,[]),key=lambda j:(0 if self.state.meta[j]["speaker"] in targets else 1,-score[j],j))
   proposed.extend(children[:2])
  for i in proposed+base:
   if len(selected)>=limit:break
   if 0<=i<len(self.ids) and i not in selected:selected.append(i)
  return selected,score
