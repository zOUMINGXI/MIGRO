"""Calibration modules; zero residual weights give an identity interface."""
import math
import torch
from torch import nn

class ResidualInterface(nn.Module):

    def __init__(self, width, rank=16):
        super().__init__()
        self.norm = nn.LayerNorm(width, elementwise_affine=False)
        self.down = nn.Linear(width, rank, bias=False)
        self.up = nn.Linear(rank, width, bias=False)
        nn.init.normal_(self.down.weight, std=1 / math.sqrt(width))
        nn.init.zeros_(self.up.weight)

    def forward(self, x):
        x = x.float()
        scale = x.square().mean(-1, keepdim=True).sqrt().detach().clamp_min(1e-07)
        return x + self.up(self.down(self.norm(x))) * scale

class CalibratedTransport(nn.Module):

    def __init__(self, base, width, rank=16):
        super().__init__()
        self.base = base
        self.adapter = ResidualInterface(width, rank)

    def forward(self, z):
        return self.adapter(self.base(z))

    def clear_query_cache(self):
        self.base.clear_query_cache()

class BoundaryAdapter(nn.Module):

    def __init__(self, width, rank=16):
        super().__init__()
        self.norm = nn.LayerNorm(width, elementwise_affine=False)
        self.down = nn.Linear(width, rank, bias=False)
        self.up = nn.Linear(rank, width, bias=False)
        nn.init.normal_(self.down.weight, std=1 / math.sqrt(width))
        nn.init.zeros_(self.up.weight)

    def forward(self, question, memory, boundary):
        q = self.down(self.norm(question.float())).mean(0)
        values = self.down(self.norm(memory.float()))
        weights = (values @ q / math.sqrt(values.shape[-1])).softmax(0)
        summary = q + (weights[:, None] * values).sum(0)
        scale = boundary.float().square().mean().sqrt().detach().clamp_min(1e-07)
        delta = 0.25 * scale * torch.tanh(self.up(torch.tanh(summary)))
        return boundary.float() + delta

class BoundaryTransport(nn.Module):

    def __init__(self, base, width):
        super().__init__()
        self.base = base
        self.adapter = BoundaryAdapter(width)

    def forward(self, z):
        return self.base(z)

    def clear_query_cache(self):
        self.base.clear_query_cache()
