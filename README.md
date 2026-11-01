# MIGRO
### Change the Model. Keep the Mind.

MIGRO studies persistent memory across language-model backbones. This initial private repository stages the performance-reporting part of a later open-source release.

## Included in this snapshot
- Checked aggregate results for 19 completed benchmark/method/model cells.
- Frozen evaluation profiles and explicit metric definitions.
- Small standalone tools for aggregating scored LoCoMo records and rendering result tables.
- A dated snapshot of evaluations that are still running.

The full memory implementation, model/interface checkpoints, benchmark inference runners and benchmark datasets are not included in this first upload. The reporting utilities can be executed independently; this is not yet an end-to-end benchmark reproduction package.

## MIGRO: complete LoCoMo
The all-question custom composite is reported on a 0–100 scale. Each model has 1,986 questions.

|Backbone|Composite|Questions|
|---|---:|---:|
|Qwen3-4B-Instruct-2507|54.13|1,986|
|Phi-3-mini-4k-instruct|49.77|1,986|
|Mistral-7B-Instruct-v0.3|45.69|1,986|

This composite was selected post hoc and is not official LoCoMo token F1. It averages the 1,540 category 1–4 question scores and 446 category 5 scores with equal weight per question. Qwen uses the AA write trajectory; Phi and Mistral use AB with their selected frozen interfaces. These entries are not a strict controlled ranking of model capability.

## MIGRO: other complete Qwen evaluations
|Benchmark|Score|Questions|Metric|
|---|---:|---:|---|
|LongMemEval-S|68.40|500|auxiliary_judge_accuracy|
|MemOps|66.30|2006|longitudinal_answer_accuracy|
|BEAM-1M|55.83|700|native_category_aggregate|

LongMemEval-S and MemOps use the recorded auxiliary judge protocol; BEAM uses category-specific scoring with auxiliary judging where required. These are research-harness results, not official leaderboard reproductions. Metrics differ across benchmarks.

See [the complete performance snapshot](results/performance_snapshot.csv), [protocol details](docs/performance_protocol.md), [snapshot scope](docs/release_scope.md) and [evaluation status](results/evaluation_status.json). Baseline results use their declared native read paths; the comparison does not equalize inference compute.

## Reporting utilities
Python 3.10 or later is sufficient; the scripts have no third-party runtime dependencies.

    python scripts/render_results.py
    python scripts/summarize_locomo.py /path/to/scored_records.json --output summary.json
    python -m unittest discover -s tests -v

The summarizer consumes already-scored records. It rejects incomplete cohorts by default and never replaces the official category-specific per-question scoring procedure. See [input format](docs/reporting_tools.md).

## Contributors
- [Mingxi Zou](https://github.com/zOUMINGXI): research lead, experiment decisions and release direction.
- [Codex](https://openai.com/codex/): AI-assisted implementation, experiment orchestration, debugging and documentation.

See [CONTRIBUTORS.md](CONTRIBUTORS.md) for attribution and commit conventions.

## Release status
Snapshot created: 2026-11-01T20:54:02.611462+08:00. Results reflect completed scored runs available at that time. Pending results have no assigned final score. Larger-read-budget candidates and storage-replica QA scores are not substituted into the frozen performance table.

This repository is currently private. A public license and full reproduction instructions will be specified when the open-source release is prepared. Third-party methods and datasets retain their respective licenses; see [third-party references](docs/third_party.md).
