"""R0/R1 fixed-route local parameter pages. Original E4 modules are untouched."""
from dataclasses import dataclass, asdict
import io, json, re
import torch
from torch.nn import functional as F

@dataclass(frozen=True)
class FixedConfig:
    capacity:int=128
    pages:int=32
    rank:int=8
    occupancy:int=4
    slots:int=192
    code_dim:int=64
    block:int=32
    seed:int=7
    ridge:float=1e-5

def canonicalize(observation):
    """Consumes observed text only. No question, answer, or hidden world fields."""
    text=observation["text"]
    m=re.fullmatch(r'On (\d{4}-\d\d-\d\d), in workspace (\S+), (\S+) reported: (\S+) has (.+?) = (.+)\.',text)
    if m:
        date,scope,source,entity,attribute,value=m.groups()
        fields=dict(source=source,scope=scope,entity=entity,attribute=attribute,value=value,effective=date,polarity="asserted")
        packet=f'Source: {source}. Scope: {scope}. Effective date: {date}. Entity: {entity}. {attribute}: {value}.'
        return packet,fields
    m=re.fullmatch(r'Workspace (\S+) recorded (\S+) on (\d{4}-\d\d-\d\d): for (\S+), the (.+?) is (.+)\.',text)
    if m:
        scope,source,date,entity,attribute,value=m.groups()
        fields=dict(source=source,scope=scope,entity=entity,attribute=attribute,value=value,effective=date,polarity="asserted")
        return f'Source: {source}. Scope: {scope}. Effective date: {date}. Entity: {entity}. {attribute}: {value}.',fields
    # No guessed coreference or lossy open-domain triple extraction.
    title=observation.get("title","document")
    return f'Source: document. Topic: {title}. Evidence: {text}',dict(source="document",residual_preserved=True)

def critical_ids(tokenizer,text):
    # Observation-only exact numerals and names; no future answer selection.
    literals=re.findall(r'\b(?:[A-Z][A-Za-z0-9_-]*|\d[\d.,:-]*)\b',text)
    return set(i for s in literals for i in tokenizer.encode(s,add_special_tokens=False))

class FixedSemanticPages:
    schema="rd-fixed-semantic-pages-v1"
    def __init__(self,core,cfg=FixedConfig()):
        self.core=core;self.cfg=cfg;self.device=core.device
        assert cfg.capacity==cfg.pages*cfg.occupancy and cfg.code_dim==core.value_dim
        self.embedding=core.codec.model.get_input_embeddings().weight.detach().float()/core.native_input_rms
        self.codebook=F.normalize(self.embedding @ core.value_basis.float(),dim=-1).T.contiguous()
        self.shape=(cfg.pages,cfg.slots*(cfg.code_dim+2),cfg.rank)
        self.reset()
    def reset(self):
        c=self.cfg;total=c.slots*(c.code_dim+2)*c.rank
        assert total%c.block==0
        self.q=torch.zeros(self.shape,device=self.device,dtype=torch.int8)
        self.scales=torch.ones((c.pages,total//c.block),device=self.device,dtype=torch.float16)
        self.curvature=torch.eye(c.rank,device=self.device)[None].repeat(c.pages,1,1)*c.ridge
        self.bank=torch.zeros(c.capacity,384,device=self.device,dtype=torch.float16)
        self.lengths=torch.zeros(c.capacity,device=self.device,dtype=torch.int16)
        self.n=0
        return self
    def metadata(self):
        return dict(schema=self.schema,cfg=asdict(self.cfg),n=self.n,core_version=self.core.core_version,
                    no_original_content_cache=True)
    def tensors(self):
        return dict(q=self.q,scales=self.scales,curvature=self.curvature,bank=self.bank,lengths=self.lengths)
    def snapshot(self):
        return dict(metadata=self.metadata(),**{k:v.detach().cpu() for k,v in self.tensors().items()})
    def restore(self,state):
        if state["metadata"]["schema"]!=self.schema or state["metadata"]["cfg"]!=asdict(self.cfg):raise ValueError("State schema/config mismatch")
        self.n=state["metadata"]["n"]
        for k in self.tensors():setattr(self,k,state[k].to(self.device))
        return self
    def ledger(self):
        out={k:v.numel()*v.element_size() for k,v in self.tensors().items()}
        out["metadata"]=len(json.dumps(self.metadata(),sort_keys=True).encode())
        out["resident_bytes"]=sum(out.values())
        b=io.BytesIO();torch.save(self.snapshot(),b);out["serialized_bytes"]=len(b.getvalue())
        return out
    @torch.no_grad()
    def feature(self,addresses,page):
        c=self.cfg
        base=F.pad(addresses.float(),(0,128))
        pos=base.new_zeros((len(base),32));pos[:,0]=1
        a=self.core.features(torch.cat([base,pos],-1))
        g=torch.Generator().manual_seed(91001+c.seed*1009+page*131)
        proj=torch.randn(self.core.width,c.rank,generator=g).to(self.device)
        return F.normalize(a@proj,dim=-1)
    def unpack(self,page):
        return (self.q[page].reshape(-1,self.cfg.block).float()*self.scales[page,:,None].float()).reshape(self.shape[1:])
    @torch.no_grad()
    def pack(self,page,w):
        blocks=w.reshape(-1,self.cfg.block)
        scales=(blocks.abs().amax(-1)/127).clamp_min(1e-7).half()
        q=(blocks/scales[:,None].float()).round().clamp(-127,127).to(torch.int8)
        self.q[page].copy_(q.reshape(self.shape[1:]));self.scales[page].copy_(scales)
    def target(self,ids,task=False,critical=None):
        values=self.embedding[ids]
        code=values@self.core.value_basis.float()
        loss0=loss1=0.
        if task:
            # Current-observation field reconstruction tasks, never evaluation QA.
            # Optimize a robust classification margin while retaining the same basis.
            with torch.enable_grad():
                original=code.detach()
                z=original.clone().requires_grad_(True)
                opt=torch.optim.Adam([z],lr=.08)
                neighbors=(F.normalize(original,dim=-1)@self.codebook).topk(16,-1).indices
                candidates=torch.cat([ids[:,None],neighbors],-1)
                duplicate=candidates[:,1:]==ids[:,None]
                cb=self.codebook.T[candidates]
                weights=torch.tensor([3. if int(i) in (critical or set()) else 1. for i in ids],device=self.device)
                for step in range(12):
                    opt.zero_grad(set_to_none=True)
                    # Deterministic bounded stress of the code, shared across arms.
                    index=torch.arange(z.numel(),device=self.device).reshape_as(z)
                    noise=((index+step*7)%19-9)/9 * original.norm(dim=-1,keepdim=True)*.015
                    logits=(F.normalize(z+noise,dim=-1)[:,None]*cb).sum(-1)*60
                    logits[:,1:]=logits[:,1:].masked_fill(duplicate,-1e4)
                    ce=F.cross_entropy(logits,torch.zeros(len(ids),device=self.device,dtype=torch.long),reduction="none")
                    loss=(ce*weights).mean()+.05*((z-original).square().sum(-1)/original.square().sum(-1).clamp_min(1e-8)).mean()
                    if step==0:loss0=float(loss)
                    loss.backward();opt.step();loss1=float(loss)
                code=z.detach()
        target=torch.cat([code,torch.ones(len(ids),1,device=self.device),values.square().mean(-1,keepdim=True)],-1)
        return F.pad(target,(0,0,0,self.cfg.slots-len(ids))),dict(task_loss_before=loss0,task_loss_after=loss1)
    def write(self,address,ids,task=False,critical=None):
        """Only current address/content is supplied. No retained original targets."""
        if self.n>=self.cfg.capacity:raise ValueError("Fixed pilot capacity exceeded; no silent eviction")
        if len(ids)>self.cfg.slots:raise ValueError("Target truncation forbidden")
        i=self.n;p=i//self.cfg.occupancy
        address=F.normalize(address[:384].float(),dim=-1).half()
        z=self.feature(address[None],p)[0]
        target,diag=self.target(ids,task,critical)
        w=self.unpack(p).detach().requires_grad_(True)
        with torch.enable_grad():
            loss=.5*(w@z-target.reshape(-1)).square().sum()
            grad=torch.autograd.grad(loss,w)[0]
        # Exact preconditioned gradient for the local quadratic, then real INT8 commit.
        c=self.curvature[p].double()+torch.outer(z.double(),z.double())
        inv=torch.linalg.inv(c).float()
        candidate=w.detach()-grad@inv
        self.pack(p,candidate)
        self.curvature[p].copy_(c.float())
        self.bank[i].copy_(address);self.lengths[i]=len(ids);self.n+=1
        diag.update(write_loss=float(loss.detach()),page=p,slot=i)
        return diag
    @torch.no_grad()
    def read_slot(self,slot):
        if slot<0 or slot>=self.n:raise IndexError(slot)
        p=slot//self.cfg.occupancy
        code=(self.unpack(p)@self.feature(self.bank[slot:slot+1],p)[0]).reshape(self.cfg.slots,-1)[:int(self.lengths[slot])]
        mass=code[:,self.cfg.code_dim:self.cfg.code_dim+1].clamp_min(.25)
        content=code[:,:self.cfg.code_dim]/mass
        logits=F.normalize(content,dim=-1)@self.codebook
        weights,ix=logits.topk(8,-1)
        weights=(weights*100).softmax(-1)
        values=(self.embedding[ix]*weights[:,:,None]).sum(1)*self.core.native_input_rms
        return values,ix[:,0],logits.gather(1,ix[:,:1]).mean()
    @torch.no_grad()
    def read_slots(self,slots):
        return torch.cat([self.read_slot(int(i))[0] for i in slots])
    @torch.no_grad()
    def route(self,query,topk=4):
        if not self.n:return []
        q=self.core.encode_semantic(["Represent this sentence for searching relevant passages: "+query,query])[:,:384]
        score=(F.normalize(q.float(),dim=-1)@self.bank[:self.n].float().T).amax(0)
        return sorted(score.topk(min(topk,self.n)).indices.tolist())
