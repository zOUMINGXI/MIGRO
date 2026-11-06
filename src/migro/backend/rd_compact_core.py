"""Drop legacy training modules unused by the frozen R2 reader."""
import gc
from types import SimpleNamespace
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
class EmbeddingOnly(nn.Module):
 def __init__(self,embedding):super().__init__();self.embedding=embedding
 def get_input_embeddings(self):return self.embedding
class CompactReadCore(nn.Module):
 @torch.no_grad()
 def encode_semantic(self, texts):
         outputs = []
         for start in range(0, len(texts), 64):
             t = self.semantic_tokenizer(
                 texts[start : start + 64],
                 padding=True,
                 truncation=True,
                 max_length=128,
                 return_tensors="pt",
             ).to(self.device)
             h = self.semantic_encoder(**t).last_hidden_state[:, 0]
             outputs.append(F.pad(F.normalize(h, dim=-1), (0, 128)))
         return torch.cat(outputs).half().float()
 def __init__(self,old):
  super().__init__()
  self.semantic_encoder=old.semantic_encoder
  self.source=EmbeddingOnly(old.codec.model.get_input_embeddings())
  self.semantic_tokenizer=old.semantic_tokenizer
  self.codec=SimpleNamespace(model=self.source,tokenizer=old.codec.tokenizer)
  self.device=old.device
  self.register_buffer("native_input_rms",old.native_input_rms.detach().clone())
  self.register_buffer("value_basis",old.value_basis.detach().clone())
  self.eval()
 def cost(self):
  return dict(parameters_bytes=sum(p.numel()*p.element_size() for p in self.parameters()),
   buffers_bytes=sum(p.numel()*p.element_size() for p in self.buffers()),
   scope="Frozen semantic encoder + source embeddings + small compatibility buffers; unused T5 transformer and previous fast-memory networks removed")

def compact_existing(old,verify_texts=()):
 before=old.encode_semantic(list(verify_texts)) if verify_texts else None
 m=CompactReadCore(old)
 if before is not None:assert torch.equal(before,m.encode_semantic(list(verify_texts)))
 assert torch.equal(old.codec.model.get_input_embeddings().weight,m.source.get_input_embeddings().weight)
 assert not any(p.requires_grad for p in m.parameters())
 return m
