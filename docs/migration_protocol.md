# Migration operations

The migration line distinguishes the operation performed by each run. Use these exact labels in result tables and figure legends.

| Operation | Existing memory reused? | Full history processed again? | Small read-interface calibration |
|---|---|---|---|
| **State Transfer** | Yes | No full-history reconstruction | Allowed; report the calibration budget |
| **Full-History Rebuilding** | No | Yes, to reconstruct target-model memory | Report separately when used |

A rebuilding run may involve memory construction or training model-specific memory components. It does not automatically imply backbone fine-tuning. The currently admitted target-native controls freeze the backbone.

Classify runs rather than making a blanket claim about a method's intrinsic capabilities. In particular, PlugMem and mem0 have both State Transfer runs and Full-History Rebuilding controls.

## Completed State Transfer results

Snapshot: 2026-11-01T22:54:24+08:00. Metric: LoCoMo all-question custom composite, 0–100, higher is better. Each result covers 1,986 questions.

| Method | Operation | Phi-3-mini-4k-instruct | Mistral-7B-Instruct-v0.3 |
|---|---|---:|---:|
| MIGRO | State Transfer | 49.77 | 45.69 |
| PlugMem | State Transfer | 36.30 | 38.60 |
| mem0 | State Transfer | 44.72 | 44.06 |

MIGRO uses existing AB memory states: a Qwen-written prefix followed by Phi continuation. Phi uses the selected step-0 identity interface; trained candidates were inspected but not selected. Mistral uses a read interface with 131,072 parameters, calibrated on 64 unique public human-QA examples for 64 optimizer updates. The backbones and shared memory core are frozen, and LoCoMo answers were not calibration targets.

PlugMem/mem0 State Transfer reuses their Qwen-built source state. These source trajectories differ from MIGRO's AB reference and must remain visible in interpretation.

## Full-History Rebuilding controls

Independent Phi/Mistral memory construction for PlugMem, mem0, TiMEM and A-Mem is labeled **Full-History Rebuilding**. PlugMem/mem0 controls provide target-built references for their transferred-state results; TiMEM/A-Mem controls report performance with memory built by the target backbone.

See [the per-run CSV](../results/migration_snapshot.csv) and [JSON](../results/migration_snapshot.json) for operation, progress, calibration and score. Pending rows have no final score.

## Unsupported configurations and reporting

LSTMem, TransMem and Doc-to-LoRA target cells that lack a matching released checkpoint remain unsupported and unscored. Their requested rebuilding operation is not evidence that reconstruction or training has been executed.

The migration-loss comparison requires a matched target-backbone rebuilding reference. Subtracting a source-backbone score from a target-backbone score does not isolate migration loss. These are previously inspected research results; the composite reporting choice is post hoc and is not official LoCoMo F1.
