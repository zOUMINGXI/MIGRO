"""Exact residual coding for semantic factor memory; no hidden target cache."""
import numpy as np

PREC=20
TOTAL=1<<PREC
LOWER=1<<23

def make_cdf(prob):
    p=np.asarray(prob,dtype=np.float64);p=p/p.sum()
    available=TOTAL-len(p);x=p*available
    freq=np.floor(x).astype(np.int64)+1
    remaining=TOTAL-int(freq.sum())
    if remaining:freq[np.argsort(-(x-np.floor(x)))[:remaining]]+=1
    c=np.concatenate([np.zeros(1,dtype=np.int64),np.cumsum(freq)])
    assert c[-1]==TOTAL and np.all(np.diff(c)>0)
    return c.astype(np.int32)

def encode(ids,cdf):
    state=LOWER;out=bytearray()
    for token in reversed(ids):
        t=int(token);start=int(cdf[t]);freq=int(cdf[t+1])-start
        limit=((LOWER>>PREC)<<8)*freq
        while state>=limit:
            out.append(state&255);state>>=8
        state=((state//freq)<<PREC)+(state%freq)+start
    return state.to_bytes(4,"little")+bytes(reversed(out))

def decode(blob,n,cdf):
    state=int.from_bytes(blob[:4],"little");offset=4;out=np.empty(n,dtype=np.int64)
    for i in range(n):
        x=state&(TOTAL-1);token=int(np.searchsorted(cdf,x,side="right"))-1
        out[i]=token;freq=int(cdf[token+1])-int(cdf[token])
        state=freq*(state>>PREC)+x-int(cdf[token])
        while state<LOWER:
            if offset>=len(blob):raise ValueError("Truncated semantic residual")
            state=(state<<8)|blob[offset];offset+=1
    if state!=LOWER or offset!=len(blob):raise ValueError("Residual length mismatch")
    return out

# Optional local native kernel; the Python implementation above is the reference.
python_encode=encode
python_decode=decode
from pathlib import Path
import ctypes
from .paths import native_dir
_lib_path=native_dir()/"libsemantic_ans.so"
NATIVE=False
if _lib_path.exists():
    _lib=ctypes.CDLL(str(_lib_path))
    _lib.ans_encode.argtypes=[ctypes.c_void_p,ctypes.c_int,ctypes.c_void_p,ctypes.c_int,ctypes.c_void_p,ctypes.c_int]
    _lib.ans_encode.restype=ctypes.c_int
    _lib.ans_decode.argtypes=[ctypes.c_void_p,ctypes.c_int,ctypes.c_int,ctypes.c_void_p,ctypes.c_int,ctypes.c_void_p]
    _lib.ans_decode.restype=ctypes.c_int
    def encode(ids,cdf):
        a=np.ascontiguousarray(ids,dtype=np.int64);c=np.ascontiguousarray(cdf,dtype=np.int32)
        out=np.empty(len(a)*4+16,dtype=np.uint8)
        n=_lib.ans_encode(a.ctypes.data,len(a),c.ctypes.data,len(c)-1,out.ctypes.data,len(out))
        if n<0:raise ValueError("ANS encoding error "+str(n))
        return out[:n].tobytes()
    def decode(blob,n,cdf):
        a=np.frombuffer(blob,dtype=np.uint8);c=np.ascontiguousarray(cdf,dtype=np.int32)
        out=np.empty(n,dtype=np.int64)
        code=_lib.ans_decode(a.ctypes.data,len(a),n,c.ctypes.data,len(c)-1,out.ctypes.data)
        if code:raise ValueError("ANS decoding error "+str(code))
        return out
    NATIVE=True

if NATIVE:
    _lib.context_encode.argtypes=[ctypes.c_void_p,ctypes.c_int,ctypes.c_void_p,ctypes.c_int,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int,ctypes.c_void_p,ctypes.c_int]
    _lib.context_encode.restype=ctypes.c_int
    _lib.context_decode.argtypes=[ctypes.c_void_p,ctypes.c_int,ctypes.c_int,ctypes.c_void_p,ctypes.c_int,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int,ctypes.c_void_p]
    _lib.context_decode.restype=ctypes.c_int
    def context_encode(ids,cdf,predictions,tables,counts):
        a=np.ascontiguousarray(ids,dtype=np.int64);out=np.empty(len(a)*8+16,dtype=np.uint8)
        n=_lib.context_encode(a.ctypes.data,len(a),cdf.ctypes.data,len(cdf)-1,predictions.ctypes.data,tables.ctypes.data,counts.ctypes.data,predictions.shape[1],out.ctypes.data,len(out))
        if n<0:raise ValueError("Context encode error "+str(n))
        return out[:n].tobytes()
    def context_decode(blob,n,cdf,predictions,tables,counts):
        a=np.frombuffer(blob,dtype=np.uint8);out=np.empty(n,dtype=np.int64)
        code=_lib.context_decode(a.ctypes.data,len(a),n,cdf.ctypes.data,len(cdf)-1,predictions.ctypes.data,tables.ctypes.data,counts.ctypes.data,predictions.shape[1],out.ctypes.data)
        if code:raise ValueError("Context decode error "+str(code))
        return out
