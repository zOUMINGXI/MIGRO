"""Opt-in lossless archive compaction over frozen Storage v1 state.

Compaction is explicit; ordinary observe/save keeps the existing v1 hot path.
IncrementalStore remains the deployment path for inexpensive durable writes.
"""
from pathlib import Path
from .storage_v1 import StorageV1
from .rd_value_packed import select as select_v1
from .rd_factorized_envelope import candidates
from .rd_semantic_root_codec import transform

PROFILE = dict(name="eviwrite-storage-v1.1-archive",new_weights=0,
 address_precision_unchanged=True,all_observations_and_versions_retained=True,
 compaction="Explicit minimum actual-byte choice; no target query or answer used",
 choices=["v1","metadata_factored","independent_root_entropy","causal_semantic_root_prediction"],
 online_commit="Use independent IncrementalStore; archive compaction is not on every write")

def select_archive(state, retained: bytes | None = None) -> tuple[bytes,dict]:
 old,old_info=select_v1(state)
 choices={"v1":old}
 envelopes,_=candidates(state);envelope=min(envelopes.values(),key=len)
 choices["metadata_factored"]=envelope
 coords=state.mem.anchor_coordinates().detach().cpu().numpy()
 for mode in ["independent","semantic"]:
  choices[mode+"_root_entropy"]=transform(envelope,coords,mode)[0]
 if retained is not None:choices["retained"]=retained
 selected=min(choices,key=lambda k:len(choices[k]))
 return choices[selected],dict(selected=selected,actual_candidate_bytes={k:len(v) for k,v in choices.items()},
  original_v1_selection=old_info,new_weight_bytes=0)

class StorageV11(StorageV1):
 def compact(self) -> dict:
  blob,selection=select_archive(self.state,self._snapshot)
  self._snapshot=blob;self._selection=selection
  return {**selection,"actual_snapshot_bytes":len(blob)}

def added_code_bytes() -> int:
 return sum((Path(__file__).parent/name).stat().st_size for name in
  ["storage_v11.py","rd_factorized_envelope.py","rd_semantic_root_codec.py"])
