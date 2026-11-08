# Selected result summaries

The repository publishes four compact groups of manuscript results.

| Summary | CSV | JSON |
|---|---|---|
| Eight methods × five primary benchmarks | [Download](performance_snapshot.csv) | [Download](performance_snapshot.json) |
| Same-host LoCoMo scale-up, Qwen 4B / 8B / 14B / 32B | [Download](quality_scaling.csv) | [Download](quality_scaling.json) |
| Qwen upgrades and cross-family migration | [Download](migration_snapshot.csv) | [Download](migration_snapshot.json) |
| Matched-state quality, storage and shared deployment | [Download](storage_summary.csv) | [Download](storage_summary.json) |

The [main README](../README.md#memory-utility) presents these results alongside continued-writing and mechanism summaries. Detailed results, task-level exports, run status and intermediate data are reserved for a later release.

Scores follow the manuscript's adopted values and metric definitions. The primary MAB macro and the separately displayed budget-sensitivity runs are distinct series. State Transfer and Full-History Rebuilding retain their per-comparison meanings. Storage quality is paired with its measured state, independently of primary benchmark quality.

Storage uses decimal MB. LoCoMo footprints sum ten conversation states; BEAM uses five million-token histories. Growth rates use the common 500K–1M window. Shared deployment estimates combine one required resource bundle with 100–100,000 independent memories.
