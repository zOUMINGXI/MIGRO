"""Continuous semantic readout and static input-vector transport.
No runtime string decoding or argmax content-token selection.
"""

import torch
from torch import nn
from torch.nn import functional as F


@torch.no_grad()
def continuous_readout(codec, query, z, max_steps=48, temperature=0.2, context_mask=None):
    model = codec.model
    embed = model.get_input_embeddings()
    dev = embed.weight.device
    header = codec.tokenizer(
        "Context: ", add_special_tokens=False, return_tensors="pt"
    ).input_ids.to(dev)
    ids = codec.tokenizer(
        "Question: " + query + "\nAnswer:", return_tensors="pt"
    ).input_ids.to(dev)
    h = (
        embed(ids)
        if z is None
        else torch.cat(
            [
                embed(header).reshape(-1, 512),
                z.to(dev).float(),
                embed(ids).reshape(-1, 512),
            ],
            0,
        )[None]
    )
    mask = torch.ones(h.shape[:2], device=dev, dtype=torch.long)
    if z is not None and context_mask is not None:
        if context_mask.numel()!=len(z):raise ValueError("Context mask length")
        mask[:,header.numel():header.numel()+len(z)]=context_mask.to(dev).long()
    enc = model.encoder(inputs_embeds=h, attention_mask=mask)
    current = embed(torch.tensor([[model.config.decoder_start_token_id]], device=dev))
    cache = None
    values = []
    entropies = []
    for _ in range(max_steps):
        out = model(
            encoder_outputs=enc,
            attention_mask=mask,
            decoder_inputs_embeds=current,
            past_key_values=cache,
            use_cache=True,
        )
        logits = out.logits[0, -1] / temperature
        probability = logits.softmax(-1)
        if probability[codec.tokenizer.eos_token_id] > 0.5:
            break
        scores, ix = logits.topk(8)
        w = scores.softmax(-1)
        current = (w[:, None] * embed.weight[ix]).sum(0)[None, None]
        values.append(current[0, 0])
        entropies.append(float(-(w * w.clamp_min(1e-9).log()).sum()))
        cache = out.past_key_values
    if not values:
        values = [torch.zeros(512, device=dev)]
    return torch.stack(values), dict(
        steps=len(values), mean_top8_entropy=sum(entropies) / max(1, len(entropies))
    )


class TransportConnector(nn.Module):
    def __init__(self, source_embeddings, host_embeddings, lexicon):
        super().__init__()
        self.register_buffer(
            "source", F.normalize(source_embeddings.detach().float(), dim=-1)
        )
        self.register_buffer("ids", lexicon["ids"].to(source_embeddings.device))
        self.register_buffer("lengths", lexicon["lengths"].to(source_embeddings.device))
        # Existing frozen host table; not a new replicated per-user buffer.
        object.__setattr__(self, "host_embeddings", host_embeddings)
        self.zero_values = False

    def forward(self, z):
        sim = F.normalize(z.float(), dim=-1) @ self.source.T
        sim, ix = sim.topk(8, dim=-1)
        w = (sim * 100.0).softmax(-1)
        lens = self.lengths[ix]
        position = torch.arange(self.ids.shape[1], device=z.device)
        valid = position[None, None, :] < lens[:, :, None]
        mass = (w[:, :, None] * valid).sum(1)
        emb = self.host_embeddings(self.ids[ix]).float()
        value = (emb * w[:, :, None, None] * valid[:, :, :, None]).sum(1) / mass[
            :, :, None
        ].clamp_min(1e-8)
        out = value[mass >= 0.5]
        return out * 0 if self.zero_values else out


def boundary_lexicon(source_tokenizer, host_tokenizer, host):
    from tokenizers import Tokenizer, normalizers

    tokenizer = Tokenizer.from_str(host_tokenizer.backend_tokenizer.to_str())
    if host == "B":
        # Phi normalizer prepends a dummy word boundary to every isolated string.
        # Source pieces already carry real word boundaries: preserve only space mapping.
        tokenizer.normalizer = normalizers.Replace(" ", "▁")
    pieces = source_tokenizer.convert_ids_to_tokens(list(range(len(source_tokenizer))))
    strings = [
        (" " if p.startswith("▁") else "")
        + source_tokenizer.decode([i], skip_special_tokens=True)
        for i, p in enumerate(pieces)
    ]
    encoded = tokenizer.encode_batch(strings, add_special_tokens=False)
    rows = [x.ids or [host_tokenizer.eos_token_id] for x in encoded]
    width = max(map(len, rows))
    ids = torch.full((len(rows), width), host_tokenizer.eos_token_id, dtype=torch.long)
    for i, row in enumerate(rows):
        ids[i, : len(row)] = torch.tensor(row)
    return dict(
        ids=ids,
        lengths=torch.tensor(list(map(len, rows))),
        source_vocab=len(rows),
        clipped_pieces=0,
        max_pieces=width,
        original_input_boundary_preserved=True,
    )


@torch.no_grad()
def read_shared_proposal(memory, state, query):
    """Read one canonical state for any host; empty state requests exact host bypass."""
    if not state.writes:
        return None, dict(empty=True, steps=0)
    q = memory.encode_semantic(
        ["Represent this sentence for searching relevant passages: " + query, query]
    )
    # Canonical FP16 transfer representation matches training/evaluation packet boundary.
    if hasattr(type(memory), "read_with_mask"):
        z,valid=memory.read_with_mask(state,q)
        z=z.masked_fill(~valid[:,None],0).detach().half()
        return continuous_readout(memory.codec,query,z,temperature=0.05,context_mask=valid)
    z = memory.read(state, q).detach().half()
    return continuous_readout(memory.codec, query, z, temperature=0.05)
