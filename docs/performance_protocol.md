# Performance evaluation protocol

## Models and selected profiles
- Qwen3-4B-Instruct-2507: source backbone for the primary completed baseline comparison.
- Phi-3-mini-4k-instruct: additional MIGRO recipient; 4K native context.
- Mistral-7B-Instruct-v0.3: additional MIGRO recipient; 32K native context.

The frozen MIGRO LoCoMo profile uses at most 2,000 source evidence tokens and 50 final answer tokens. Qwen uses its bound reader. Phi uses its selected boundary interface and option-blind answer policy; Mistral uses its selected 64-example calibrated interface. Backbones and selected interfaces are frozen during evaluation. The historical AA/AB write trajectories differ across hosts.

## LoCoMo metric
All 1,986 official question IDs are retained. Categories 1–4 contain 1,540 questions; category 5 contains 446.

Composite = (1540 × mean_category_1_to_4_token_F1  + 446 × mean_category_5_score)  / 1986.

The components in this formula are on a 0–100 scale. The composite is equivalent to averaging per-question scores on a common scale. It is a custom post-hoc reporting choice, not official LoCoMo F1. Category-specific normalization, date handling and category 5 decoding are performed by the upstream scorer before aggregation.

The repository shows one final LoCoMo metric. Raw benchmark labels, conversations and predictions are not redistributed in this initial snapshot.

## Migration operations

| Operation | Meaning |
|---|---|
| **State Transfer** | Reuse an existing persistent memory state on another backbone. Small read-interface calibration is allowed. |
| **Full-History Rebuilding** | Process the complete raw history again to reconstruct target-model memory and any required model-specific memory components. |

Operation is a property of each run. A method that supports State Transfer can also have a Full-History Rebuilding control. Backbone training is recorded separately.

For MIGRO, the Phi result uses the selected step-0 identity interface; the Mistral result uses a 131,072-parameter read interface calibrated on 64 unique public human-QA examples (64 optimizer updates). Both backbones and the shared memory core remain frozen. The stored memory is transferred rather than rebuilt for these rows.

See [the migration protocol](migration_protocol.md) for trajectories and [the labeled snapshot](../results/migration_snapshot.csv) for the selected operation-labeled comparisons.

## Baseline execution differences
- PlugMem retains its native planning/reasoning stages.
- mem0, TiMEM and A-Mem retain their declared memory/retrieval ports and final evidence budgets.
- LSTmem uses its retained-history profile and native released Qwen adapter.
- TransMem uses its released Qwen checkpoint, native context suffix and official overflow mechanism.
- Doc-to-LoRA uses the separately named native-window 1/N stabilized variant. This was a post-hoc repair for repeated-output collapse and is not the vanilla published preset. Historical abnormal 8192-chunk scores are excluded from the snapshot.
- Engineering ports and bounded response-format repairs are part of the recorded harness. Shared construction, embedding and auxiliary-serving choices differ from original published configurations.

These methods do not all use equal inference compute or context windows. Their scores must be interpreted as declared operating points.

## Other benchmarks
- LongMemEval-S: 500 questions, auxiliary gpt-5.4-mini judging, selected frozen Qwen reader; accuracy on a 0–100 scale.
- MemOps: 2,006 longitudinal operation questions; the snapshot shows answer accuracy. Operation detection, state accuracy, leakage and other diagnostics are separate quantities.
- BEAM-1M: 700 questions across 35 histories, category-specific native score aggregation and auxiliary judging where required. The completed MIGRO run has 100 previously inspected questions; it is not an entirely unseen confirmation set.

Output budgets differ by benchmark: LoCoMo 50, LongMemEval 256, MemOps 1024 and BEAM 768 tokens under the recorded settings.

## Selection and uncertainty
This is a frozen research snapshot after historical development and interface selection. It does not correct for all prior choices. No statistical-significance or quality-constrained Pareto claim follows solely from these aggregate tables.

The detailed experiments retain their matched question IDs and appropriate conversation/history/identity clustering. Bootstrap intervals and full per-question scoring artifacts are not reimplemented by the initial reporting utility.

## Published summaries

The [selected result files](../results/README.md) follow the current manuscript's primary benchmark, scaling, migration and storage tables. MemoryAgentBench reports a four-competency macro in the primary table; the README's separate matched-budget series retains its own reader/budget operating points. Detailed task-level results and intermediate artifacts are reserved for a later release.
