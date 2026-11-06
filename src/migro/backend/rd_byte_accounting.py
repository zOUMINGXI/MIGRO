import sys
from .rd_rate_v2 import decode_blob

def deep_bytes(v,seen=None):
 if seen is None:seen=set()
 if id(v) in seen:return 0
 seen.add(id(v));n=sys.getsizeof(v)
 if isinstance(v,dict):return n+sum(deep_bytes(k,seen)+deep_bytes(x,seen) for k,x in v.items())
 if isinstance(v,(list,tuple)):return n+sum(deep_bytes(x,seen) for x in v)
 return n
def actual_online(blob):
 meta,t=decode_blob(blob,"cpu")
 if meta["schema"].startswith("semantic-"):
  raw=t["binding_bytes"].numpy().tobytes();ends=t["binding_ends"].tolist()
  blocks=[raw[a:b] for a,b in zip(ends,ends[1:])]
  eraw=t["exception_bytes"].numpy().tobytes();eend=t["exception_ends"].tolist()
  ex={i:eraw[a:b] for i,a,b in zip(t["exception_indices"].tolist(),eend,eend[1:])}
  py=deep_bytes([blocks,t["binding_modes"].tolist(),t["lengths"].tolist(),ex])
  tensors=sum(t[k].numel()*t[k].element_size() for k in ["keys","scales","topics"])
 else:
  raw=t["binding_bytes"].numpy().tobytes();ends=t["binding_ends"].tolist()
  blocks=[raw[a:b] for a,b in zip(ends,ends[1:])]
  py=deep_bytes([blocks,t["binding_lengths"].tolist()])
  tensors=sum(x.numel()*x.element_size() for k,x in t.items() if not k.startswith("binding_"))
 return dict(online_persistent_tensor_bytes=tensors,online_python_containers_and_bindings_bytes=py,
  online_persistent_state_bytes=tensors+py,online_scope="Tensor payload plus actual resident binding/list/dict objects; excludes public models, allocator bookkeeping and temporary serialization")
