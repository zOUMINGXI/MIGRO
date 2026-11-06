"""Actual owned reader objects and immutable table sharing for new experiments."""
import sys
import numpy as np,torch
_PUBLIC={}
def share_public_tables(memory):
 base=(id(memory.core),str(memory.device),str(memory.cfg.model_path),str(memory.cfg.basis_path))
 names=("global_cdf","predictions","context_tables","context_counts","offsets","cdf_offsets","mean","full_basis","axis_scale","weights")
 for name in names:
  if not hasattr(memory,name):continue
  key=base+(name,str(getattr(memory.cfg,"weights_path","")) if name=="weights" else "")
  if key not in _PUBLIC:_PUBLIC[key]=getattr(memory,name)
  else:setattr(memory,name,_PUBLIC[key])
 return memory
def owned_bytes(value,seen=None,storages=None):
 if seen is None:seen=set()
 if storages is None:storages=set()
 if id(value) in seen:return 0
 seen.add(id(value))
 if isinstance(value,torch.Tensor):
  key=(str(value.device),value.untyped_storage().data_ptr())
  if key in storages:return sys.getsizeof(value)
  storages.add(key);return sys.getsizeof(value)+value.untyped_storage().nbytes()
 if isinstance(value,np.ndarray):
  # Most reader arrays own their allocation; count a view's backing owner once.
  return sys.getsizeof(value)+(owned_bytes(value.base,seen,storages) if value.base is not None else 0)
 n=sys.getsizeof(value)
 if isinstance(value,dict):return n+sum(owned_bytes(k,seen,storages)+owned_bytes(v,seen,storages) for k,v in value.items())
 if isinstance(value,(list,tuple,set,frozenset)):return n+sum(owned_bytes(v,seen,storages) for v in value)
 return n
def reader_ledger(reader):
 fields={k:v for k,v in vars(reader).items() if k not in ("state","core","mem","ranker","stem") and not callable(v)}
 return dict(reader_owned_cpu_and_tensor_bytes=owned_bytes(fields),dense_address_tensor_bytes=reader.keys.numel()*reader.keys.element_size(),
  decoded_reader_cache_present=True,scope="Owned decoded IDs, strings, terms, ranking arrays and dense address tensor; excludes shared models and compressed state; CPython containers counted")
