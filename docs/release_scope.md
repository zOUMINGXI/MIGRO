# Initial release scope

Included:
- 19 complete scored aggregate entries with explicit benchmark, model, metric and profile labels.
- MIGRO LoCoMo on three backbones; Qwen baseline LoCoMo/LongMemEval-S comparisons; completed MIGRO Qwen MemOps and BEAM.
- Status-only rows for pending Qwen and native-target evaluations, captured at the snapshot time.
- Independently runnable reporting code and its targeted tests.

The full memory model, inference runners, interfaces and experiment orchestration are planned for a later release. This first repository upload must not be described as a complete open-source implementation or a full benchmark reproduction.

Current native-target controls extend LoCoMo to Phi and Mistral. PlugMem/mem0 shared-source-state target reads and independently target-built controls have distinct roles. Their results are not mixed. A cross-model score difference is not automatically a target-relative migration loss.

The checked-in status JSON is a dated snapshot, not a live dashboard. Unsupported checkpoints and pending evaluations receive no fabricated zero score.
