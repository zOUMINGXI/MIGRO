# Initial release scope

Included:
- 19 complete scored aggregate entries with explicit benchmark, model, metric and profile labels.
- MIGRO LoCoMo on three backbones; Qwen baseline LoCoMo/LongMemEval-S comparisons; completed MIGRO Qwen MemOps and BEAM.
- Status-only rows for pending Qwen and native-target evaluations, captured at the snapshot time.
- Independently runnable reporting code and its targeted tests.

The full memory model, inference runners, interfaces and experiment orchestration are planned for a later release. This first repository upload must not be described as a complete open-source implementation or a full benchmark reproduction.

LoCoMo target-model runs are labeled by operation: **State Transfer** reuses existing memory, with small interface calibration permitted; **Full-History Rebuilding** reprocesses the full history to construct target-model memory. MIGRO, PlugMem and mem0 have completed State Transfer rows. PlugMem/mem0 also have rebuilding controls, alongside the target-built TiMEM/A-Mem rows. These operations are not mixed. See [migration protocol](migration_protocol.md) and [labeled results](../results/migration_snapshot.csv). A cross-model score difference is not automatically a target-relative migration loss.

The checked-in status JSON is a dated snapshot, not a live dashboard. Unsupported checkpoints and pending evaluations receive no fabricated zero score.
