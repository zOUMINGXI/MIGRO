# Third-party notices

MIGRO-owned source code, interface tensors, projection bases and coding tables are
distributed under Apache-2.0. Copyright 2026 Mingxi Zou.

The shared core contains separately licensed pretrained components. Their original
licenses continue to apply to those components.

| Component | Original source | License | Distributed portion |
|---|---|---|---|
| BGE-small-en-v1.5 | https://huggingface.co/BAAI/bge-small-en-v1.5 | MIT | Frozen encoder weights, model configuration and tokenizer |
| FLAN-T5-small | https://huggingface.co/google/flan-t5-small | Apache-2.0 | Frozen input embedding matrix and tokenizer |
| MIGRO | https://github.com/zOUMINGXI/MIGRO | Apache-2.0 | Projection bases, coding tables and frozen read interfaces |

The BGE license and its copyright notice are reproduced in licenses/BGE-MIT.txt.
The full Apache-2.0 terms are in LICENSE. PROVENANCE.json records the component
mapping and upstream revisions used for this bundle.

The core export retains the BGE encoder and FLAN-T5 input embeddings. The other
FLAN-T5 transformer layers are omitted. Interface checkpoints contain their adapter
tensors and resolved runtime scope. Training configurations, optimizer data and
experiment bookkeeping have been removed. Tensor values and dtypes are preserved.

## Models obtained separately

| Runtime dependency | Original source | Declared license |
|---|---|---|
| Qwen3-4B-Instruct-2507 | https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507 | Apache-2.0 |
| Mistral-7B-Instruct-v0.3 | https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3 | Apache-2.0 |
| Phi-3-mini-4k-instruct | https://huggingface.co/microsoft/Phi-3-mini-4k-instruct | MIT |
| MS MARCO MiniLM-L6-v2 reranker | https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2 | Apache-2.0 |

These external model weights are downloaded from their respective publishers.
Their model cards, licenses and applicable usage conditions remain authoritative.
Python dependencies are installed separately under their own licenses.
