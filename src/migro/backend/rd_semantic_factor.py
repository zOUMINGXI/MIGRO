from .paths import asset_path
"""Shared semantic factors for both routing and exact binding prediction.

This is a semantic-latent / explicit-residual hybrid. Coded residual bytes are
charged. It does not claim that entropy-coded words are pure parametric facts.
"""
import json,zlib,time
from pathlib import Path
from dataclasses import dataclass,asdict
import numpy as np
import torch
from torch.nn import functional as F
from .rd_rate_v2 import encode_blob,decode_blob
from .rd_semantic_ans import encode as ans_encode,decode as ans_decode,context_encode,context_decode,TOTAL

@dataclass(frozen=True)
class FactorConfig:
    key_mode:str="factor128"
    semantic_bindings:bool=True
    candidates:int=128
    block:int=32
    model_path:str="research/rd_semantic_factor_20261025/model/codec.npz"

class FactorBindings:
    def __init__(self,owner):
        self.owner=owner;self.blocks=[];self.modes=[];self.lengths=[];self.max_scratch=0
    def decode(self,i):
        c=self.owner.cfg.block;p=i//c;j=i%c;start=p*c;n=min(c,len(self.lengths)-start)
        b=self.blocks[p];mode=self.modes[p]
        if mode==0:
            raw=np.frombuffer(zlib.decompress(b),dtype="<u2");a=sum(self.lengths[start:i])
            return raw[a:a+self.lengths[i]].astype(np.int64)
        lengths=np.frombuffer(b[:2*n],dtype="<u2");a=2*n+int(lengths[:j].sum())
        topic=int(self.owner.topics[i]) if mode in (2,4) else 0
        cdf=self.owner.cdfs[topic] if mode in (2,4) else self.owner.global_cdf
        piece=b[a:a+int(lengths[j])]
        if mode in (3,4):return context_decode(piece,self.lengths[i],cdf,self.owner.predictions,self.owner.context_tables,self.owner.context_counts)
        return ans_decode(piece,self.lengths[i],cdf)
    def append(self,ids):
        c=self.owner.cfg.block;start=(len(self.lengths)//c)*c
        old=[self.decode(i) for i in range(start,len(self.lengths))]
        values=old+[np.asarray(ids,dtype=np.int64)]
        raw=np.concatenate(values).astype("<u2").tobytes();choices=[(zlib.compress(raw,6),0)]
        for mode in ([1,2,3,4] if self.owner.cfg.semantic_bindings else [1,3]):
            pieces=[]
            for j,v in enumerate(values):
                topic=int(self.owner.topics[start+j]) if mode in (2,4) else 0
                cdf=self.owner.cdfs[topic] if mode in (2,4) else self.owner.global_cdf
                pieces.append(context_encode(v,cdf,self.owner.predictions,self.owner.context_tables,self.owner.context_counts) if mode in (3,4) else ans_encode(v,cdf))
            b=np.asarray([len(p) for p in pieces],dtype="<u2").tobytes()+b"".join(pieces)
            choices.append((b,mode))
        chosen,mode=min(choices,key=lambda p:len(p[0]))
        if len(self.lengths)%c:self.blocks[-1]=chosen;self.modes[-1]=mode
        else:self.blocks.append(chosen);self.modes.append(mode)
        self.lengths.append(len(ids));self.max_scratch=max(self.max_scratch,len(raw))
    def tensors(self):
        return dict(binding_bytes=torch.from_numpy(np.frombuffer(b"".join(self.blocks),dtype=np.uint8).copy()),
            binding_ends=torch.from_numpy(np.cumsum([0]+[len(b) for b in self.blocks],dtype=np.int32)),
            binding_modes=torch.tensor(self.modes,dtype=torch.uint8),
            lengths=torch.tensor(self.lengths,dtype=torch.int16))
    def restore(self,t):
        ends=t["binding_ends"].cpu().tolist();raw=t["binding_bytes"].cpu().numpy().tobytes()
        self.blocks=[raw[a:b] for a,b in zip(ends,ends[1:])]
        self.modes=t["binding_modes"].cpu().tolist();self.lengths=t["lengths"].cpu().tolist()

class SemanticFactorMemory:
    def __init__(self,core,cfg=FactorConfig()):
        self.core=core;self.cfg=cfg;self.device=core.device
        d=np.load(asset_path(cfg.model_path))
        self.cdfs=np.ascontiguousarray(d["cdfs"],dtype=np.int32);self.global_cdf=np.ascontiguousarray(d["global_cdf"],dtype=np.int32)
        self.predictions=np.ascontiguousarray(d["predictions"],dtype=np.int16)
        self.context_tables=np.ascontiguousarray(d["context_tables"],dtype=np.int32)
        self.context_counts=np.ascontiguousarray(d["context_counts"],dtype=np.uint8)
        self.mean=torch.from_numpy(d["mean"]).to(self.device)
        self.basis=torch.from_numpy(d["basis"]).to(self.device)
        self.centers=torch.from_numpy(d["centers"]).to(self.device)
        self.projected_centers=(self.centers-self.mean)@self.basis
        width=128 if cfg.key_mode=="factor128" else 384
        self.keys=torch.empty(0,width,device=self.device,dtype=torch.float16 if cfg.key_mode=="full16" else torch.int8)
        self.scales=torch.empty(0,device=self.device,dtype=torch.float16)
        self.topics=torch.empty(0,device=self.device,dtype=torch.uint8)
        self.bindings=FactorBindings(self);self.exceptions={};self.n=0;self.last_route_stats={}
    def choose_topic(self,key,ids):
        cc=(key@self.centers.T).topk(8).indices.cpu().numpy()
        frequencies=self.cdfs[cc[:,None],ids[None,:]+1]-self.cdfs[cc[:,None],ids[None,:]]
        costs=(-np.log2(frequencies.astype(np.float64)/TOTAL)).sum(1)
        return int(cc[costs.argmin()])
    def _semantic_ids(self,text):
        return self.core.semantic_tokenizer(text,truncation=True,max_length=128)["input_ids"]
    def write(self,address,ids,text):
        ids=ids.detach().cpu().numpy().astype(np.int64)
        key=F.normalize(address[:384].float().to(self.device),dim=-1)
        topic=self.choose_topic(key,ids)
        self.topics=torch.cat([self.topics,torch.tensor([topic],device=self.device,dtype=torch.uint8)])
        self.bindings.append(ids)
        decoded=self.core.codec.tokenizer.decode(ids.tolist(),skip_special_tokens=True)
        original=self._semantic_ids(text)
        if self.cfg.key_mode=="factor128" and self._semantic_ids(decoded)!=original:
            self.exceptions[self.n]=zlib.compress(np.asarray(original,dtype="<i4").tobytes(),6)
        if self.cfg.key_mode=="factor128":
            target=(key-self.mean)@self.basis-self.projected_centers[topic]
            # Current-observation semantic coefficients are the only local target.
            with torch.enable_grad():
                w=torch.zeros_like(target,requires_grad=True);loss=.5*(w-target).square().sum()
                value=(w-torch.autograd.grad(loss,w)[0]).detach()
        else:value=key;loss=torch.tensor(0.)
        if self.cfg.key_mode=="full16":stored=value.half()
        else:
            scale=(value.abs().max()/127).clamp_min(1e-7).half()
            stored=(value/scale.float()).round().clamp(-127,127).to(torch.int8)
            self.scales=torch.cat([self.scales,scale[None]])
        self.keys=torch.cat([self.keys,stored[None]]);self.n+=1
        return dict(write_loss=float(loss),binding_mode=self.bindings.modes[-1],topic=topic)
    def read_slot(self,i,values=True):
        ids=torch.from_numpy(self.bindings.decode(i)).to(self.device)
        return (self.core.codec.model.get_input_embeddings()(ids),ids) if values else ids
    def read_slots(self,slots):
        return torch.cat([self.read_slot(i)[0] for i in slots]) if slots else None
    @torch.no_grad()
    def _exact_keys(self,slots):
        inputs=[]
        for i in slots:
            if i in self.exceptions:ids=np.frombuffer(zlib.decompress(self.exceptions[i]),dtype="<i4").tolist()
            else:ids=self._semantic_ids(self.core.codec.tokenizer.decode(self.bindings.decode(i).tolist(),skip_special_tokens=True))
            inputs.append(ids)
        out=[]
        for j in range(0,len(inputs),64):
            batch=inputs[j:j+64];n=max(map(len,batch))
            ids=torch.zeros(len(batch),n,device=self.device,dtype=torch.long);mask=torch.zeros_like(ids)
            for k,v in enumerate(batch):ids[k,:len(v)]=torch.tensor(v,device=self.device);mask[k,:len(v)]=1
            h=self.core.semantic_encoder(input_ids=ids,attention_mask=mask,token_type_ids=torch.zeros_like(ids)).last_hidden_state[:,0]
            out.append(F.normalize(F.normalize(h,dim=-1).half().float(),dim=-1).half())
        return torch.cat(out)
    @torch.no_grad()
    def route(self,query,topk=4,candidates=None):
        q=self.core.encode_semantic(["Represent this sentence for searching relevant passages: "+query,query])[:,:384]
        q=F.normalize(q.float(),dim=-1);self.last_route_stats={}
        if self.cfg.key_mode=="factor128":
            approximate=self.keys.float()*self.scales.float()[:,None]+self.projected_centers[self.topics.long()]
            score=(q@self.basis@approximate.T+ (q@self.mean)[:,None]).amax(0)
            selected=score.topk(min(self.cfg.candidates if candidates is None else candidates,self.n)).indices.tolist()
            exact=self._exact_keys(selected)
            s=(q@exact.float().T).amax(0);ix=s.topk(min(topk,len(selected))).indices.tolist()
            self.last_route_stats=dict(reranked=len(selected),source_history_loaded=False,values_reconstructed_from_current_state=True)
            return sorted(selected[j] for j in ix)
        bank=self.keys.float()
        if self.cfg.key_mode=="full8":bank=bank*self.scales.float()[:,None]
        return sorted((q@bank.T).amax(0).topk(min(topk,self.n)).indices.tolist())
    def tensors(self):
        t=dict(keys=self.keys,scales=self.scales,topics=self.topics);t.update(self.bindings.tensors())
        order=sorted(self.exceptions);pieces=[self.exceptions[i] for i in order]
        t.update(exception_indices=torch.tensor(order,dtype=torch.int32),
         exception_ends=torch.from_numpy(np.cumsum([0]+[len(b) for b in pieces],dtype=np.int32)),
         exception_bytes=torch.from_numpy(np.frombuffer(b"".join(pieces),dtype=np.uint8).copy()))
        return t
    def metadata(self):
        return dict(schema="semantic-factor-v1",config=asdict(self.cfg),n=self.n,
         representation="shared semantic-latent coefficients plus explicit lossless binding residual",
         original_text_cache=False,original_history_read=False,candidate_memory_reencoding=self.cfg.key_mode=="factor128")
    def serialize(self):return encode_blob(self.metadata(),self.tensors())
    def restore(self,blob):
        meta,t=decode_blob(blob,self.device);assert meta["config"]==asdict(self.cfg)
        self.n=meta["n"]
        for k in ("keys","scales","topics"):setattr(self,k,t[k])
        self.bindings.restore(t);self.exceptions={}
        ix=t["exception_indices"].cpu().tolist();ends=t["exception_ends"].cpu().tolist();raw=t["exception_bytes"].cpu().numpy().tobytes()
        self.exceptions={i:raw[a:b] for i,a,b in zip(ix,ends,ends[1:])}
        return self
    def ledger(self):
        t=self.tensors();size=lambda names:sum(t[k].numel()*t[k].element_size() for k in names)
        return dict(serialized_bytes=len(self.serialize()),tensor_payload_bytes=size(t.keys()),
         directory_bytes=size(["keys","scales","topics"]),
         binding_bytes=size(["binding_bytes","binding_ends","binding_modes","lengths"]),
         exact_encoder_exception_bytes=size(["exception_indices","exception_ends","exception_bytes"]),
         exceptions=len(self.exceptions),semantic_blocks=sum(m in (2,4) for m in self.bindings.modes),
         global_entropy_blocks=sum(m in (1,3) for m in self.bindings.modes),zlib_blocks=self.bindings.modes.count(0),binding_mode_counts={str(m):self.bindings.modes.count(m) for m in range(5)},
         shared_codec_file_bytes=asset_path(self.cfg.model_path).stat().st_size,
         shared_codec_runtime_array_bytes=sum(x.nbytes for x in [self.cdfs,self.global_cdf,self.predictions,self.context_tables,self.context_counts])+sum(x.numel()*x.element_size() for x in [self.mean,self.basis,self.centers,self.projected_centers]),
         transient_binding_block_max_bytes=self.bindings.max_scratch)
