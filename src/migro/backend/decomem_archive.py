"""DeCoMem exact neural content archive for existing portable memory states.

Keeps learned geometry, source-token bindings, metadata and versions unchanged.
The predictor is a shared public dependency, not part of each user's state.
"""
import json,struct
import numpy as np
from .causal_content_codec import compress,decompress
from .causal_neural_content import NeuralContent,unpack
from .rd_value_packed import geometry_blob,ArrayBindings,ValuePackedState

MAGIC=b"DCMARC01"

def encode(state,coder,progress=None):
    rows=[];exceptions={}
    tok=state.core.codec.tokenizer
    for i,meta in enumerate(state.meta):
        ids=state.mem.read_slot(i,False).cpu().tolist()
        text=tok.decode(ids,skip_special_tokens=False)
        if tok.encode(text,add_special_tokens=False)!=ids:exceptions[str(i)]=ids
        rows.append(dict(body=text,scope=str(meta.get("session","default"))))
    if not rows:raise ValueError("Use the literal archive for an empty state")
    archives,cost=coder.encode_prefixes(rows,[len(rows)],progress=progress)
    content=archives[len(rows)]
    geometry=compress(geometry_blob(state),"zstd19")
    escape=compress(json.dumps(exceptions,separators=(",",":")).encode(),"zstd19")
    blob=MAGIC+struct.pack("<QQQ",len(geometry),len(content),len(escape))+geometry+content+escape
    return blob,dict(bytes=len(blob),geometry_bytes=len(geometry),content_bytes=len(content),
                     source_id_escape_bytes=len(escape),source_id_escape_rows=len(exceptions),**cost)

def restore(core,blob,coder=None):
    from .rd_robust_residual import restore_any
    if not blob.startswith(MAGIC):raise ValueError("Not a DeCoMem neural archive")
    ng,nc,ne=struct.unpack_from("<QQQ",blob,8);assert len(blob)==32+ng+nc+ne
    geometry=blob[32:32+ng];content=blob[32+ng:32+ng+nc];escape=blob[-ne:]
    base=restore_any(core,decompress(geometry,"zstd19"))
    if coder is None:
        h,_=unpack(content);coder=NeuralContent(batch=h["batch"],page_size=h["page_size"])
    rows,_=coder.decode(content);exceptions=json.loads(decompress(escape,"zstd19"))
    assert len(rows)==base.mem.n
    values=[np.asarray(exceptions.get(str(i),core.codec.tokenizer.encode(r["body"],add_special_tokens=False)),dtype="<u2") for i,r in enumerate(rows)]
    flat=np.concatenate(values) if values else np.empty(0,dtype="<u2")
    base.mem.bindings=ArrayBindings(flat,np.asarray([len(v) for v in values],dtype="<u4"))
    return ValuePackedState(base)
