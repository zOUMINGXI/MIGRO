# Migration operations

| Operation | Existing memory reused? | Full history processed again? | Read-interface calibration |
|---|---|---|---|
| State Transfer | Yes | No full-history reconstruction | Allowed under the declared budget |
| Full-History Rebuilding | No | Yes | Recorded separately when used |

The primary comparisons use State Transfer for MIGRO, PlugMem and Mem0. TiMem, A-MEM, LSTMem, TransMem and Doc-to-LoRA use target-side rebuilding. Backbone fine-tuning is separate from memory construction or refitting.

## Receiving-host results

The Qwen chain follows 4B → 8B → 14B → 32B. The first cross-family path follows Qwen 4B → Mistral 7B → Phi. LoCoMo reports the custom all-question composite.

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

LSTMem, TransMem and Doc-to-LoRA have completed Phi/Mistral rebuilding results. The separate Ministral 14B → Gemma 26B-A4B extension retains pending entries where the manuscript has no final score.

[Selected migration CSV](../results/migration_snapshot.csv) · [JSON](../results/migration_snapshot.json) · [Same-host scale-up](../results/quality_scaling.csv)

## Interfaces and trajectories

The retained MIGRO Phi/Mistral reference uses the AB memory trajectory: a Qwen-written prefix followed by Phi continuation. Phi uses the selected step-0 identity interface. Mistral uses a 131,072-parameter read interface calibrated on 64 unique public human-QA examples for 64 optimizer updates. The backbones and shared memory core are frozen, and LoCoMo answers are not calibration targets.

PlugMem and Mem0 transfer their Qwen-built source states. A migration-loss comparison requires a matched target-backbone rebuilding reference. The chain stage and execution mode identify each summary row; rebuilding does not carry source-host state.

The second heterogeneous chain reuses the same immutable Qwen-built state across Ministral and Gemma, with no writes between receiving stages and no new calibration examples. Gemma 26B-A4B denotes total/active MoE size.

## Continued writing and mechanisms

The [README summaries](../README.md#keep-writing-after-the-handoff) report continued writing under Mistral against final-history labels. Target-built prefixes and rebuilding on append are distinguished from inherited-state continuation.

Local mechanism controls use separately evaluated full-interface states. They compare direct reuse against simple history access and target rebuilding, then remove boundary composition while holding the evidence and host fixed.
