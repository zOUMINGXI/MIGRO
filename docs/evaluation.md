# Experimental evidence

An upgrade should retain useful history and leave room for the next experience. The current results test memory utility, larger hosts, continued writing and persistent footprint. Completed cohorts are shown below; the [dated result index](../results/README.md) records protocols and pending work.

### Memory utility

| Method | LoCoMo | LongMemEval-S | MemOps | BEAM-1M | MAB · primary |
|---|---:|---:|---:|---:|---:|
| MIGRO | 54.13 | 68.40 | 66.30 | 55.83 | 35.16 |
| PlugMem | 41.89 | 43.20 | 35.09 | 34.11 | 7.76 |
| Mem0 | 52.12 | 56.80 | 52.13 | 52.43 | 32.82 |
| TiMem | 49.13 | 48.40 | 50.66 | 51.90 | 18.90 |
| A-MEM | 36.20 | 35.10 | 39.78 | 33.25 | 16.72 |
| LSTMem | 40.64 | 26.00 | 52.59 | 28.44 | 6.56 |
| TransMem | 45.85 | 37.40 | 62.51 | 28.76 | 22.80 |
| Doc-to-LoRA | 14.96 | 12.80 | 15.35 | 30.20 | 11.62 |

Scores are on a 0–100 scale. LoCoMo uses the custom all-question composite (1,986 questions); LongMemEval-S uses auxiliary accuracy (500); MemOps uses longitudinal answer accuracy (2,006); BEAM uses its category aggregate (700). MAB uses a custom four-ability macro (2,071). These metrics are not interchangeable.

**Separate MAB budget-sensitivity runs:** at the same 2,000-source-token cap, MIGRO scores **35.14** and TransMem **22.88**; at 8,000, **40.30** and **28.99**. These retained budget-sensitivity runs are separate from the manuscript's primary MAB table. Other methods retain their native read paths; equal source caps do not equalize inference compute.

[Benchmark summary · CSV](../results/performance_snapshot.csv) · [JSON](../results/performance_snapshot.json) · [Scoring definitions](../docs/performance_protocol.md)

### Larger hosts, existing memory

|Receiving Qwen host|MIGRO · State Transfer from 4B|MIGRO · Full-History Rebuilding|
|---|---:|---:|
|8B|55.52|55.68|
|14B|60.37|60.32|
|32B|62.33|61.83|

The separate heterogeneous chain **Qwen 4B → Ministral 14B → Gemma 26B-A4B** scores **57.41 → 66.28** on the two receiving hosts, with a 2,000-source-token budget and no new calibration examples. Both hops reuse the same immutable state, with no writes between hops. Gemma 26B-A4B denotes total/active MoE size. This chain differs from the historical Phi/Mistral experiment below.

[Same-host scaling](../results/quality_scaling.csv) · [Migration summaries](../results/migration_snapshot.csv)

<details>
<summary><strong>All methods · same-host scale-up and Qwen migration</strong></summary>

| Host | MIGRO | PlugMem | Mem0 | TiMem | A-MEM | LSTMem | TransMem | Doc-to-LoRA |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen3-4B-Instruct-2507 | 54.13 | 41.89 | 52.12 | 49.13 | 36.20 | 40.64 | 45.85 | 14.96 |
| Qwen3-8B | 55.68 | 34.86 | 54.88 | 46.19 | 44.70 | 19.70 | 46.39 | 12.77 |
| Qwen3-14B | 60.32 | 40.10 | 54.36 | 50.25 | 40.88 | 16.58 | 49.83 | 12.30 |
| Qwen3-32B | 61.83 | 44.66 | 57.16 | 49.24 | 45.38 | 14.35 | 48.03 | 13.18 |

| Method | Operation | Qwen 8B | Qwen 14B | Qwen 32B |
|---|---:|---:|---:|---:|
| MIGRO | State Transfer | 55.52 | 60.37 | 62.33 |
| PlugMem | State Transfer | 44.41 | 46.10 | 49.15 |
| Mem0 | State Transfer | 53.18 | 57.07 | 57.17 |
| TiMem | Full-History Rebuilding | 46.19 | 50.25 | 49.24 |
| A-MEM | Full-History Rebuilding | 44.70 | 40.88 | 45.38 |
| LSTMem | Full-History Rebuilding | 19.70 | 16.58 | 14.35 |
| TransMem | Full-History Rebuilding | 46.39 | 49.83 | 48.03 |
| Doc-to-LoRA | Full-History Rebuilding | 12.77 | 12.30 | 13.18 |

TransMem's Qwen rebuilding values are the same-host references used in the manuscript's transfer profile.

</details>

### Keep writing after the handoff

In the Qwen 4B → Mistral 7B continuation study, MIGRO scores **45.13** after appending the second half, versus **36.94** with transfer alone and **45.19** with full-history target rebuilding. All three are scored against the same final history. Old-history, new-history and cross-boundary questions are reported separately for all eight methods.

<details>
<summary><strong>Continued-writing results across all methods</strong></summary>

| Method | First-half memory only | Continued writing | Full-history rebuilding |
|---|---:|---:|---:|
| MIGRO | 36.94 | 45.13 | 45.19 |
| PlugMem | 33.46 | 37.38 | 36.95 |
| Mem0 | 35.59 | 43.89 | 44.37 |
| TiMem | 32.30 | 37.17 | 40.92 |
| A-MEM | 29.67 | 34.01 | 37.87 |
| LSTMem | 11.34 | 11.15 | 11.16 |
| TransMem | 9.60 | 7.04 | 7.04 |
| Doc-to-LoRA | 11.04 | 10.98 | 11.32 |

All conditions use final-history labels under the Mistral receiver. LSTMem and Doc-to-LoRA use target-built prefixes; TransMem rebuilds its view on append. The other methods extend inherited prefix states.

</details>

### Access across model families

The manuscript follows Qwen 4B → Mistral 7B → Phi. Receiving-host LoCoMo scores use the operation shown for each method.

| Method | Operation | Mistral 7B | Phi |
|---|---:|---:|---:|
| MIGRO | State Transfer | 45.69 | 49.77 |
| PlugMem | State Transfer | 38.60 | 36.30 |
| Mem0 | State Transfer | 44.06 | 44.72 |
| TiMem | Full-History Rebuilding | 41.22 | 28.03 |
| A-MEM | Full-History Rebuilding | 37.73 | 29.57 |
| LSTMem | Full-History Rebuilding | 11.31 | 16.84 |
| TransMem | Full-History Rebuilding | 7.04 | 18.79 |
| Doc-to-LoRA | Full-History Rebuilding | 11.32 | 14.07 |

[Migration summary · CSV](../results/migration_snapshot.csv) · [JSON](../results/migration_snapshot.json) · [Operations and interfaces](../docs/migration_protocol.md)

<details>
<summary><strong>Evaluation details and interpretation</strong></summary>

The LoCoMo composite was selected post hoc. It weights all 1,986 questions equally: 1,540 category 1–4 question scores and 446 category 5 scores. It is not official LoCoMo token F1. The host rows do not form a strict controlled ranking of model capability.

For MIGRO's AB reference, the existing state contains a Qwen-written prefix followed by Phi continuation. Phi uses the selected step-0 identity interface. Mistral uses a read interface with 131,072 parameters, calibrated on 64 unique public human-QA examples for 64 optimizer updates. The backbones and shared memory core are frozen; LoCoMo answers are not calibration targets.

PlugMem and mem0 reuse their Qwen-built source states for their State Transfer rows. Their source trajectories differ from MIGRO's AB reference. A migration-loss claim requires a matched target-backbone rebuilding reference.

LongMemEval-S and MemOps use the recorded auxiliary judge protocol. BEAM uses category-specific scoring with auxiliary judging where required. These are research-harness evaluations, not official leaderboard reproductions. Baselines retain their declared native read paths; inference compute is not equalized.

</details>

### Compact memory, economical growth

Quality and bytes are paired on the same evaluated states. LoCoMo storage sums ten conversation states; BEAM storage is per million-token history.

| Method | LoCoMo | Private MB / 10 states | BEAM MB / history | 500K–1M bytes/token |
|---|---:|---:|---:|---:|
| MIGRO | 54.13 | 1.115 | 1.857 | 1.809 |
| PlugMem | 41.89 | 92.078 | 82.871 | 79.608 |
| Mem0 | 50.45 | 160.671 | 125.499 | 123.093 |
| TiMem | 47.22 | 10.851 | 6.166 | 6.178 |
| A-MEM | 35.29 | 23.340 | 23.191 | 23.639 |
| LSTMem | 40.64 | 1.871 | 4.371 | 4.370 |
| TransMem | 45.96 | 1.200 | 7.092 | 7.778 |
| Doc-to-LoRA | 14.96 | 283.140 | 56.625 | 28.313 |

MIGRO averages **1.857 MB** per million-token history. Sharing its required resources across **1,000 independent memories** gives **2.154 MB per memory**, including fixed deployment assets. The common growth window is **500K–1M source tokens**.

<details>
<summary><strong>Shared-resource estimates · 100–100,000 independent memories</strong></summary>

Average MB per million-token memory, including required assets shared once:

| Method | 100 memories | 1,000 | 10,000 | 100,000 |
|---|---:|---:|---:|---:|
| MIGRO | 4.829 | 2.154 | 1.887 | 1.860 |
| PlugMem | 84.237 | 83.008 | 82.885 | 82.873 |
| Mem0 | 127.018 | 125.651 | 125.514 | 125.500 |
| TiMem | 7.532 | 6.303 | 6.180 | 6.167 |
| A-MEM | 24.557 | 23.328 | 23.205 | 23.193 |
| LSTMem | 4.775 | 4.411 | 4.375 | 4.371 |
| TransMem | 24.292 | 8.812 | 7.264 | 7.109 |
| Doc-to-LoRA | 73.465 | 58.309 | 56.794 | 56.642 |

</details>

[Storage summary · CSV](../results/storage_summary.csv) · [JSON](../results/storage_summary.json)

### Mechanisms under examination

The experiment contracts separate questions about the complete system from questions about individual components.

|Research question|Controlled comparison|Specification|
|---|---|---|
|Does the memory system provide useful evidence?|Full memory, BM25, recent history and no-memory controls, with target-native rebuilding as a reference.|[Core transfer](../configs/ablations/core_transfer.json)|
|What does the host interface contribute?|Boundary composition crossed with frozen calibration.|[Transport and calibration](../configs/ablations/transport_composition.json)|
|What is the address space–quality tradeoff?|Anchor + residual, anchor-only and original full addresses.|[Address representation](../configs/ablations/addresses.json)|
|What does persistence coding save?|Full lossless selection, existing literal candidates and an uncompressed archive.|[Persistence encoding](../configs/ablations/persistence.json)|

The manuscript reports the following matched controls. LoCoMo values use the local full-mechanism reference of 54.21, separately from the primary table's 54.13.

| Mechanism | Full mechanism | Simpler control |
|---|---:|---:|
| Evidence selection | 54.21 | 50.48 without host association; 44.82 semantic Top-K |
| Semantic addresses | 54.21 · Anchor + Residual | 51.73 · anchor only |
| Mistral boundary composition | 45.48 | 39.98 without boundary composition |
| Archive encoding | 0.870 MB | 0.953 MB existing lossless; 2.381 MB uncompressed |

Anchor + Residual reduces address-table bytes by **91.1%** relative to float32, with a **0.46-point** quality gap. Core transfer controls also outperform BM25, recent-history windows and no memory; the reported direct-minus-BM25 gains are **11.74 points on Phi** and **3.93 on Mistral**.
