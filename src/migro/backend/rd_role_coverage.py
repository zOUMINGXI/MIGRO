"""Role-sensitive evidence ranking with the same 64 pairs and 2000-token budget."""
import re
import numpy as np
from .rd_bound_evidence_reader import BoundEvidenceReader
class RoleCoverageReader(BoundEvidenceReader):
 def candidates(self,query,stratified=True,limit=64):
  result=super().candidates(query,stratified,limit);self._current_candidates=result[0];return result
 def select_evidence(self,query,topk=16,budget=2000,mode="support"):
  # Query wording selects a prior, never a hard filter or gold label.
  assistant=bool(re.search(r"\byou (?:gave|suggested|recommended|mentioned|said|told|provided)|\byour (?:advice|recommendation|suggestion|example)|\b(?:assistant|you) (?:previously )?(?:recommend|suggest)",query,re.I))
  own=bool(re.search(r"\b(?:I|my|me|mine|our|we)\b",query,re.I))
  target="assistant" if assistant else "user" if own else None
  ranker=self.ranker;captured={};reader=self
  class PriorRanker:
   def score(self,q,passages):
    raw=ranker.score(q,passages)
    slots=reader._current_candidates
    assert len(slots)==len(passages)
    bias=np.asarray([.75 if target is not None and reader.state.meta[i]["speaker"].lower()==target else 0. for i in slots])
    captured.update(raw=raw.tolist(),bias=bias.tolist())
    return raw+bias
  try:
   self.ranker=PriorRanker()
   z,text,meta=super().select_evidence(query,topk,budget,mode)
  finally:self.ranker=ranker
  meta.update(role_prior_target=target,role_prior_weight=.75,unadjusted_candidate_scores=captured["raw"],candidate_role_bias=captured["bias"],
   no_speaker_exclusion=True,candidate_pair_budget_unchanged=True)
  return z,text,meta
