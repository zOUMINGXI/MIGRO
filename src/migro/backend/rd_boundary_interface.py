"""Evidence-conditioned residual at the assistant boundary; native evidence/question tokens stay exact."""
import math
import torch
from torch import nn
from .host_bridge import MARKER
from .rd_query_interface import question_mask

class BoundaryAdapter(nn.Module):
    def __init__(self,width,rank=16):
        super().__init__()
        self.norm=nn.LayerNorm(width,elementwise_affine=False)
        self.down=nn.Linear(width,rank,bias=False)
        self.up=nn.Linear(rank,width,bias=False)
        nn.init.normal_(self.down.weight,std=1/math.sqrt(width))
        nn.init.zeros_(self.up.weight)
    def forward(self,question,memory,boundary):
        q=self.down(self.norm(question.float())).mean(0)
        values=self.down(self.norm(memory.float()))
        weights=(values@q/math.sqrt(values.shape[-1])).softmax(0)
        summary=q+(weights[:,None]*values).sum(0)
        scale=boundary.float().square().mean().sqrt().detach().clamp_min(1e-7)
        # Bound the perturbation while leaving all observed and question vectors intact.
        delta=.25*scale*torch.tanh(self.up(torch.tanh(summary)))
        return boundary.float()+delta

class BoundaryTransport(nn.Module):
    def __init__(self,base,width):
        super().__init__();self.base=base;self.adapter=BoundaryAdapter(width)
    def forward(self,z):return self.base(z)
    def clear_query_cache(self):self.base.clear_query_cache()

def build_boundary_item(bridge,row,z):
    emb=bridge.host.get_input_embeddings()
    before,after=bridge.render(row,soft=True).split(MARKER)
    left=torch.tensor(bridge.ids(before),device=bridge.device)
    right_ids,mask=question_mask(bridge,row,after)
    right=torch.tensor(right_ids,device=bridge.device)
    values=bridge.connector.base(z.to(bridge.device)).to(emb.weight.dtype)
    tail=emb(right)
    boundary=bridge.connector.adapter(tail[mask],values,tail[-1]).to(tail.dtype)
    x=torch.cat([emb(left),values,tail])
    x=x.clone();x[-1]=boundary
    labels=torch.full((len(x),),-100,device=bridge.device,dtype=torch.long)
    return x,labels,dict(prompt_tokens=len(left)+len(right),soft_tokens=len(values),
        answer_tokens=0,sequence_tokens=len(x),memory_tokens_adapted=0,
        native_question_tokens_adapted=0,assistant_boundary_tokens_adapted=1)
