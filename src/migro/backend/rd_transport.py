"""Transport vocabulary-domain repair, isolated from all historical modules."""
import torch
from torch.nn import functional as F
from .continuous_readout import TransportConnector

class ValidVocabularyTransport(TransportConnector):
    def forward(self,z):
        sim=F.normalize(z.float(),dim=-1)@self.source.T
        # Model embeddings can contain padding rows beyond the tokenizer vocabulary.
        # Such rows have no token semantics or lexicon entry and must not be candidates.
        sim[:,len(self.lengths):]=-torch.inf
        sim,ix=sim.topk(8,dim=-1)
        w=(sim*100.).softmax(-1)
        lens=self.lengths[ix]
        position=torch.arange(self.ids.shape[1],device=z.device)
        valid=position[None,None,:]<lens[:,:,None]
        mass=(w[:,:,None]*valid).sum(1)
        emb=self.host_embeddings(self.ids[ix]).float()
        value=(emb*w[:,:,None,None]*valid[:,:,:,None]).sum(1)/mass[:,:,None].clamp_min(1e-8)
        out=value[mass>=.5]
        return out*0 if self.zero_values else out
