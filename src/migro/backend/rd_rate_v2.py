"""R2 causal page-rate controls. The R1/E4 code and results remain unchanged."""
import io,json,struct,time
from dataclasses import dataclass,asdict
import numpy as np
import torch
from torch.nn import functional as F
from .rd_semantic import canonicalize

@dataclass(frozen=True)
class RateConfig:
    dim:int=64
    rank:int=8
    occupancy:int=4
    precision:str="8"
    seed:int=7
    block:int=32
    ridge:float=1e-5
    target:str="canonical"
    policy:str="unbounded"
    cap_bytes:int=3145728

def compact_canonical(observation):
    _,f=canonicalize(observation)
    if f.get("residual_preserved"):
        # Natural source residual remains verbatim; no guessed fact extraction.
        return observation["text"]
    return (f'{f["effective"]}, {f["scope"]}, {f["source"]}: '
            f'{f["entity"]}; {f["attribute"]}: {f["value"]}.')

def pack_weights(w,bits,block=32):
    if bits==16:return w.half().contiguous(),torch.empty(0,device=w.device,dtype=torch.float16)
    flat=w.flatten();pad=(-flat.numel())%block
    blocks=F.pad(flat,(0,pad)).reshape(-1,block)
    limit=7 if bits==4 else 127
    scales=(blocks.abs().amax(-1)/limit).clamp_min(1e-7).half()
    signed=(blocks/scales[:,None].float()).round().clamp(-limit,limit).to(torch.int8).flatten()
    if bits==8:return signed,scales
    unsigned=(signed.to(torch.int16)+8).to(torch.uint8)
    return (unsigned[::2]|(unsigned[1::2]<<4)),scales

def unpack_weights(q,scales,bits,shape,block=32):
    if bits==16:return q.float().reshape(shape)
    if bits==4:
        u=torch.stack([q&15,q>>4],-1).flatten().to(torch.int16)-8
    else:u=q
    flat=(u.float().reshape(-1,block)*scales.float()[:,None]).flatten()
    return flat[:int(np.prod(shape))].reshape(shape)

def encode_blob(meta,tensors):
    specs=[];chunks=[]
    for name,t in tensors.items():
        a=t.detach().cpu().contiguous().numpy()
        specs.append(dict(name=name,dtype=a.dtype.str,shape=list(a.shape),bytes=a.nbytes))
        chunks.append(a.tobytes())
    header=json.dumps(dict(meta=meta,tensors=specs),separators=(",",":")).encode()
    return b"EVRD2001"+struct.pack("<I",len(header))+header+b"".join(chunks)

def encoded_size(meta,tensors):
    types={torch.int8:"|i1",torch.uint8:"|u1",torch.int16:"<i2",torch.int32:"<i4",torch.float16:"<f2",torch.float32:"<f4"}
    specs=[dict(name=name,dtype=types[t.dtype],shape=list(t.shape),bytes=t.numel()*t.element_size()) for name,t in tensors.items()]
    header=json.dumps(dict(meta=meta,tensors=specs),separators=(",",":")).encode()
    return 12+len(header)+sum(s["bytes"] for s in specs)

def decode_blob(blob,device):
    assert blob[:8]==b"EVRD2001"
    size=struct.unpack("<I",blob[8:12])[0];header=json.loads(blob[12:12+size]);off=12+size;tensors={}
    for s in header["tensors"]:
        a=np.frombuffer(blob[off:off+s["bytes"]],dtype=s["dtype"]).reshape(s["shape"]).copy();off+=s["bytes"]
        tensors[s["name"]]=torch.from_numpy(a).to(device)
    assert off==len(blob)
    return header["meta"],tensors

class RatePages:
    """Actual local parameter pages, no content/target cache in persistent state.

    RECODE consumes only previously decoded page values plus the current observation.
    Adaptive selection verifies all existing decoded tokens before committing. External
    prefix audits against original observations independently test cumulative damage.
    """
    def __init__(self,core,cfg=RateConfig()):
        self.core=core;self.cfg=cfg;self.device=core.device
        self.embedding=core.codec.model.get_input_embeddings().weight.detach().float()/core.native_input_rms
        self.basis=core.value_basis[:,:cfg.dim].float()
        self.codebook=F.normalize(self.embedding@self.basis,dim=-1).T.contiguous()
        self.pages=[];self.bank=torch.empty(0,384,device=self.device,dtype=torch.float16)
        self.lengths=torch.empty(0,device=self.device,dtype=torch.int16)
        self.n=0;self.attempted=0;self.rejected=0
        self.source_indices=torch.empty(0,device=self.device,dtype=torch.int32)
        self.max_workspace_bytes=0
    @torch.no_grad()
    def features(self,keys,page):
        base=F.pad(keys.float(),(0,128));pos=base.new_zeros((len(base),32));pos[:,0]=1
        a=self.core.features(torch.cat([base,pos],-1))
        g=torch.Generator().manual_seed(91001+self.cfg.seed*1009+page*131)
        proj=torch.randn(self.core.width,self.cfg.rank,generator=g).to(self.device)
        return F.normalize(a@proj,dim=-1)
    def decode_codes(self,codes):
        content=codes[...,:self.cfg.dim]/codes[...,self.cfg.dim:self.cfg.dim+1].clamp_min(.25)
        return F.normalize(content,dim=-1)@self.codebook
    @torch.no_grad()
    def read_slot(self,slot,values=True):
        p=slot//self.cfg.occupancy;page=self.pages[p]
        w=unpack_weights(page["q"],page["scale"],page["bits"],page["shape"],self.cfg.block)
        codes=(w@self.features(self.bank[slot:slot+1],p)[0]).reshape(-1,self.cfg.dim+2)[:int(self.lengths[slot])]
        logits=self.decode_codes(codes)
        if not values:return logits.argmax(-1)
        logits,ids=logits.topk(8,-1);weights=(logits*100).softmax(-1)
        z=(self.embedding[ids]*weights[:,:,None]).sum(1)*self.core.native_input_rms
        return z,ids[:,0]
    @torch.no_grad()
    def read_slots(self,slots):
        kept=[self.physical_slot(i) for i in slots]
        kept=[i for i in kept if i is not None]
        return torch.cat([self.read_slot(i)[0] for i in kept]) if kept else None
    def physical_slot(self,logical):
        if self.cfg.policy=="unbounded":return logical if 0<=logical<self.n else None
        positions=(self.source_indices==logical).nonzero().flatten()
        return int(positions[0]) if len(positions) else None
    def tensors(self):
        t=dict(bank=self.bank,lengths=self.lengths)
        if self.cfg.policy!="unbounded":t["source_indices"]=self.source_indices
        for i,p in enumerate(self.pages):
            for k in ("q","scale","curvature"):t[f"p{i}/{k}"]=p[k]
        return t
    def metadata(self):
        return dict(schema="rate-pages-v2-logical-budget-ids",config=asdict(self.cfg),n=self.n,attempted=self.attempted,rejected=self.rejected,
                    pages=[dict(bits=p["bits"],shape=p["shape"]) for p in self.pages],
                    persistent_original_targets=False)
    def serialize(self):return encode_blob(self.metadata(),self.tensors())
    def restore(self,blob):
        meta,t=decode_blob(blob,self.device)
        assert meta["config"]==asdict(self.cfg)
        self.n=meta["n"];self.attempted=meta["attempted"];self.rejected=meta["rejected"]
        self.bank=t["bank"];self.lengths=t["lengths"];self.pages=[]
        self.source_indices=t.get("source_indices",torch.empty(0,device=self.device,dtype=torch.int32))
        for i,p in enumerate(meta["pages"]):
            self.pages.append({**p,**{k:t[f"p{i}/{k}"] for k in ("q","scale","curvature")}})
        return self
    def ledger(self):
        tensors=self.tensors()
        payload=sum(x.numel()*x.element_size() for x in tensors.values())
        meta=len(json.dumps(self.metadata(),separators=(",",":")).encode())
        return dict(tensor_bytes=payload,metadata_bytes=meta,resident_bytes=payload+meta,
                    serialized_bytes=len(self.serialize()),
                    index_bytes=self.bank.numel()*self.bank.element_size()+self.lengths.numel()*2+(self.source_indices.numel()*4 if self.cfg.policy!="unbounded" else 0),
                    page_bytes=sum(p["q"].numel()*p["q"].element_size()+p["scale"].numel()*2 for p in self.pages),
                    curvature_bytes=sum(p["curvature"].numel()*4 for p in self.pages),
                    tracked_workspace_tensor_bytes_partial=self.max_workspace_bytes)
    def limit(self):
        if self.cfg.policy=="fixed3":return self.cfg.cap_bytes
        if self.cfg.policy=="sqrt":return int(self.cfg.cap_bytes*(max(1,self.attempted)/256)**.5)
        return None
    def write(self,address,ids):
        """Writer arguments are current observation only; future questions unavailable."""
        self.attempted+=1;c=self.cfg;p=self.n//c.occupancy;start=p*c.occupancy
        old=[self.read_slot(i,values=False) for i in range(start,self.n)]
        ids=ids.to(self.device);targets=old+[ids];lengths=[len(x) for x in targets]
        keys=torch.cat([self.bank[start:],F.normalize(address[:384].float(),dim=-1).half()[None]])
        z=self.features(keys,p);slots=max(lengths)
        y=[]
        for tokens in targets:
            e=self.embedding[tokens]
            code=torch.cat([e@self.basis,torch.ones(len(e),1,device=self.device),e.square().mean(-1,keepdim=True)],-1)
            y.append(F.pad(code,(0,0,0,slots-len(tokens))).flatten())
        y=torch.stack(y,1)
        # Local parameter fitting is an actual preconditioned observation-loss gradient.
        # Earlier information comes exclusively from this memory's pre-write decoding.
        curvature=z.T.double()@z.double()+torch.eye(c.rank,device=self.device,dtype=torch.float64)*c.ridge
        with torch.enable_grad():
            w=torch.zeros(y.shape[0],c.rank,device=self.device,requires_grad=True)
            loss=.5*(w@z.T-y).square().sum()
            grad=torch.autograd.grad(loss,w)[0]
            candidate=(-grad.double()@torch.linalg.inv(curvature)).float()
        choices=[4,8,16] if c.precision=="adaptive" else [int(c.precision)]
        chosen=None;matches=[];attempts=0
        for bits in choices:
            attempts+=1;q,s=pack_weights(candidate,bits,c.block)
            restored=unpack_weights(q,s,bits,list(candidate.shape),c.block)
            reconstructed=(restored@z.T).T.reshape(len(targets),slots,c.dim+2)
            predicted=self.decode_codes(reconstructed.reshape(-1,c.dim+2)).argmax(-1).reshape(len(targets),slots)
            matches=[bool(torch.equal(predicted[i,:n],targets[i])) for i,n in enumerate(lengths)]
            chosen=dict(q=q,scale=s,bits=bits,shape=list(candidate.shape),curvature=curvature.float())
            if all(matches):break
        if c.precision=="adaptive" and not all(matches):
            raise RuntimeError("Even FP16 failed exact token preservation; allocate capacity/revise representation, do not claim pass.")
        self.max_workspace_bytes=max(self.max_workspace_bytes,
            sum(t.numel()*t.element_size() for t in [keys,z,y,grad,candidate,restored,reconstructed,predicted,curvature])+sum(t.numel()*t.element_size() for t in targets))
        oldpage=self.pages[p] if p<len(self.pages) else None
        if oldpage is None:self.pages.append(chosen)
        else:self.pages[p]=chosen
        oldbank,oldlengths,oldsource=self.bank,self.lengths,self.source_indices
        self.bank=torch.cat([self.bank,keys[-1:]])
        self.lengths=torch.cat([self.lengths,torch.tensor([len(ids)],device=self.device,dtype=torch.int16)])
        self.n+=1
        if c.policy!="unbounded":self.source_indices=torch.cat([self.source_indices,torch.tensor([self.attempted-1],device=self.device,dtype=torch.int32)])
        limit=self.limit()
        if limit is not None and encoded_size(self.metadata(),self.tensors())>limit:
            self.bank,self.lengths,self.source_indices=oldbank,oldlengths,oldsource;self.n-=1;self.rejected+=1
            if oldpage is None:self.pages.pop()
            else:self.pages[p]=oldpage
            return dict(action="REJECT_BUDGET",accepted=False)
        return dict(action="RECODE_"+str(chosen["bits"]),accepted=True,bits=chosen["bits"],candidate_trials=attempts,
                    page_exact=all(matches),write_loss=float(loss.detach()))
    @torch.no_grad()
    def route(self,query,topk=4):
        if self.n==0:return []
        q=self.core.encode_semantic(["Represent this sentence for searching relevant passages: "+query,query])[:,:384]
        score=(F.normalize(q.float(),dim=-1)@self.bank.float().T).amax(0)
        physical=score.topk(min(topk,self.n)).indices
        if self.cfg.policy!="unbounded":return sorted(self.source_indices[physical].tolist())
        return sorted(physical.tolist())

class RateKV:
    """Strong explicit token KV with the identical keys, targets and soft interface."""
    def __init__(self,core,int8_keys=False):
        self.core=core;self.device=core.device;self.int8_keys=int8_keys
        self.bank=torch.empty(0,384,device=self.device,dtype=torch.int8 if int8_keys else torch.float16)
        self.scales=torch.empty(0,device=self.device,dtype=torch.float16)
        self.ids=torch.empty(0,device=self.device,dtype=torch.int16)
        self.offsets=torch.zeros(1,device=self.device,dtype=torch.int32);self.n=0
    def write(self,address,ids):
        key=F.normalize(address[:384].float(),dim=-1)
        if self.int8_keys:
            s=(key.abs().max()/127).clamp_min(1e-7).half()
            key=(key/s.float()).round().clamp(-127,127).to(torch.int8)
            self.scales=torch.cat([self.scales,s[None]])
        else:key=key.half()
        assert int(ids.max())<32768
        self.bank=torch.cat([self.bank,key[None]]);self.ids=torch.cat([self.ids,ids.to(self.device).short()])
        self.offsets=torch.cat([self.offsets,torch.tensor([len(self.ids)],device=self.device,dtype=torch.int32)]);self.n+=1
        return dict(action="KV_WRITE",accepted=True)
    def read_slot(self,i,values=True):
        ids=self.ids[int(self.offsets[i]):int(self.offsets[i+1])].long()
        return (self.core.codec.model.get_input_embeddings()(ids),ids) if values else ids
    def read_slots(self,slots):
        return torch.cat([self.read_slot(i)[0] for i in slots]) if slots else None
    def tensors(self):return dict(bank=self.bank,scales=self.scales,ids=self.ids,offsets=self.offsets)
    def metadata(self):return dict(schema="rate-kv-v1",n=self.n,int8_keys=self.int8_keys,values="int16 token IDs")
    def serialize(self):return encode_blob(self.metadata(),self.tensors())
    def restore(self,blob):
        meta,t=decode_blob(blob,self.device);assert meta["int8_keys"]==self.int8_keys
        self.n=meta["n"]
        for k,v in t.items():setattr(self,k,v)
        return self
    def ledger(self):
        t=self.tensors();payload=sum(x.numel()*x.element_size() for x in t.values());meta=len(json.dumps(self.metadata(),separators=(",",":")).encode())
        return dict(tensor_bytes=payload,metadata_bytes=meta,resident_bytes=payload+meta,serialized_bytes=len(self.serialize()),
                    index_bytes=self.bank.numel()*self.bank.element_size()+self.scales.numel()*2+self.offsets.numel()*4,
                    page_bytes=self.ids.numel()*2,curvature_bytes=0,tracked_workspace_tensor_bytes_partial=0)
    @torch.no_grad()
    def route(self,query,topk=4):
        q=self.core.encode_semantic(["Represent this sentence for searching relevant passages: "+query,query])[:,:384]
        bank=self.bank.float()
        if self.int8_keys:bank=bank*self.scales.float()[:,None]
        score=(F.normalize(q.float(),dim=-1)@bank.T).amax(0)
        return sorted(score.topk(min(topk,self.n)).indices.tolist())
