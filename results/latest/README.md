# Current experiment results

Snapshot: 2026-11-08T03:15:51.047178+00:00

- [Performance](../performance_snapshot.csv): all eight methods on LoCoMo, LongMemEval-S, MemOps and BEAM, including the author-reported Table 1 results.
- [MAB](memoryagentbench.csv), [task and ability scores](memoryagentbench_tasks.json): 130 contexts / 2,071 questions per entry. The four-ability macro is a project summary, not an official aggregate. Original task metrics are retained with format-only answer extraction; auxiliary judging uses gpt-5.4-mini. MIGRO and TransMem have matched 2,000 and 8,000 observed-source-token caps; other methods retain native read paths. Equal caps do not imply equal FLOPs.
- [LoCoMo scale-up](locomo_scaleup.csv): native rebuilding and 4B state transfer are separate. MIGRO 32B uses the previously accepted two_round_v4 reader; its original values remain in original_score.
- [Original three-host migration](../migration_snapshot.csv): historical MIGRO AB trajectories are retained. [New heterogeneous scale-up](heterogeneous_scaleup.csv) separates sequential MIGRO handoff, baseline source-state transfer, target-side rebuilding, and pending adapter qualification.
- [Continued writing](continued_writing.csv): all eight methods, three conditions, old/new/cross/all groups. The receiver is Mistral-7B, not Phi. Every arm uses final full-history labels. Cross means annotated evidence in both halves, not a validated revision label; cross_multihop is a subset of cross.
- [Storage](storage/): same-state quality and byte accounting, long-history growth, and shared deployment scenarios for 100–100,000 independent memories.
- Mechanism ablations: [performance](ablations/performance/REPORT.md), [migration](ablations/migration/REPORT.md), [storage](ablations/storage/REPORT.md), with full aggregate JSON alongside each report. Fresh paired controls are distinct from the historical primary table.
- [Recorded latency audit](latency/README.md): six overlapping timing dimensions, with missingness and available coverage; no experiment rerun.
- [Pending status](status.json): dated snapshot, not a live dashboard.

## Operation labels

Continued-writing MIGRO, mem0, PlugMem, TiMEM and A-Mem reuse source state in this specific implementation. LSTmem and Doc-to-LoRA rebuild the target prefix before appending. TransMem reconstructs its retained-history view on append; this arm equals rebuilding, not persistent-state transfer. Internal transfer_* column names are retained for compatibility; condition and operation give their actual meaning.

LoCoMo reports the custom all-question composite, not official token F1. Research-harness baseline ports include declared target refits and the stabilized Doc-to-LoRA mean-chunk variant. Read docs/performance_protocol.md and docs/migration_protocol.md with these tables.

## Storage interpretation

Private state excludes separately shared deployment assets and the common host model/runtime. The first six methods use the existing physical restorable-file study; parametric additions use uncompressed native tensors/text and operational metadata. TransMem includes its retained raw suffix. Ephemeral recomputable KV caches are excluded.

The parametric LoCoMo quality rows come from reloaded measured states; they do not replace the main performance table. BEAM uses five fixed natural histories, with actual token-boundary overshoot. Local 975k–1M and late-average 500k–1M marginal windows are different quantities. Shared cost uses F + N*S and F/N + S; these are arithmetic storage scenarios, not concurrent-serving measurements. Doc-to-LoRA radar coverage is complete; its optional dense curve was stopped with 93/283 points measured.

## Heterogeneous chain protocol

MIGRO hands off the same immutable Qwen-built state through Ministral-3-14B-Instruct-2512-BF16 and google/gemma-4-26B-A4B-it. There are no writes between hops, no new calibration and no backbone training. This is portability of existing state, not continued learning between hops. Gemma 26B-A4B denotes total/active MoE size, not 26B active compute.
