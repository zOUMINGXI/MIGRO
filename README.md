<p align="center">
  <img src="assets/migro-launch.svg" alt="MIGRO — Models evolve. Memory endures. Persistent memory across language models." width="1600">
</p>

<p align="center">
  <strong>Persistent memory across language models</strong><br>
  A shared substrate for accumulated experience. Adaptable interfaces for the models that use it.
</p>

<p align="center">
  <a href="#the-continuity-problem">Overview</a> &nbsp; · &nbsp;
  <a href="#system-design">Architecture</a> &nbsp; · &nbsp;
  <a href="#experimental-evidence">Evidence</a> &nbsp; · &nbsp;
  <a href="#released-components">Code</a> &nbsp; · &nbsp;
  <a href="docs/system_overview.md">System notes</a> &nbsp; · &nbsp;
  <a href="docs/performance_protocol.md">Evaluation protocol</a>
</p>

---

An agent's model can change in an afternoon. Its useful history may take months to accumulate.

That history carries the context behind decisions: what a user values, which constraints still apply, how a plan changed, and what happened before the current conversation. As models evolve, preserving access to that experience becomes a systems problem.

**MIGRO is a memory architecture for carrying accumulated experience across language-model hosts.** Its central design separates a persistent memory substrate from the interfaces through which individual backbones read it. Writing, storage and cross-model access are treated as parts of the same continuity problem.

> **The ambition: let improvements in the model build on the experience already accumulated by the agent.**

## The continuity problem

Long-lived agents operate on a different timescale from the models that power them. A deployment may change backbone, tokenizer or model family while its history continues to grow. Each change raises a practical question: **what must be carried forward, and what must be rebuilt?**

MIGRO approaches that question at three levels:

|System concern|Design objective|
|---|---|
|**Memory utility**|Recover evidence from accumulated history and use it to answer questions.|
|**Persistent footprint**|Represent content and retrieval addresses compactly, with explicit accounting for stored state and shared deployment assets.|
|**Host continuity**|Reconnect existing state to a receiving backbone through a host-specific read interface.|

These objectives shape both the architecture and the evaluation. A smaller archive must still restore the intended state. A successful host connection must still provide useful evidence. A migration comparison must identify the operation actually performed.

## System design

<p align="center">
  <img src="assets/migro-system.svg" alt="MIGRO reference architecture: observed experience enters a source-host writer; persistent memory stores evidence, semantic addresses and encoded state; a transport and calibration interface connects it to receiving hosts." width="1600">
</p>

### A persistent substrate with adaptable host connections

MIGRO separates the stored experience from the machinery used to expose it to a particular host.

**Writing preserves a connection to observed history.** Native reference writing uses the source host to construct memory from observations. Persistent payloads and metadata retain the relationship between stored content and its originating experience.

**Addressing and encoding serve different roles.** Semantic addresses support evidence selection. The anchor-and-residual representation is evaluated against anchor-only and full-address controls. Persistence encoding is evaluated separately, including exact restoration of source IDs, address geometry and metadata.

**The read interface handles the receiving host.** Lexical transport connects source representations to the target vocabulary and embedding interface. Calibration adapts the host connection where the protocol permits it. This factorization makes host-specific adaptation an explicit part of the system.

The diagram describes the research architecture. The current code release exposes selected components of that architecture; the complete state backend and checkpoints are outside this upload.

[Architecture and component contracts](docs/system_overview.md) · [Implementation details](docs/implementation.md) · [Ablation configurations](configs/ablations)

### Carry state or rebuild history

The migration protocol distinguishes two concrete operations:

|Operation|Persistent state|Target-side work|
|---|---|---|
|**State Transfer**|Reuse existing memory state.|Connect it to the receiving host; small read-interface calibration is permitted and reported separately.|
|**Full-History Rebuilding**|Construct memory again from the full raw history.|Run the target-side construction process, including model-specific memory training when required.|

The label belongs to the run. A method can have both transferred-state results and independently rebuilt target controls. Rebuilding memory does not, by itself, mean fine-tuning the backbone.

[Migration operations and calibration budgets](docs/migration_protocol.md)

## Experimental evidence

The checked-in results examine memory use within a host and access across model families. Tables below retain the frozen repository snapshot; metric definitions and source trajectories are part of their interpretation.

### Memory across conversations and long histories

|Evaluation|MIGRO score|Questions|Reported metric|
|---|---:|---:|---|
|**LoCoMo** · Qwen3-4B-Instruct-2507|**54.13**|1,986|All-question custom composite|
|**LongMemEval-S**|**68.40**|500|Auxiliary judge accuracy|
|**MemOps**|**66.30**|2,006|Longitudinal answer accuracy|
|**BEAM-1M**|**55.83**|700|Native category aggregate|

**Each row uses its recorded benchmark-specific protocol.** LoCoMo reports a custom composite on a 0–100 scale, not official token F1. The other benchmarks use their respective scoring procedures; their scores are not interchangeable.

[Complete performance snapshot](results/performance_snapshot.csv) · [Scoring definitions and evaluation profiles](docs/performance_protocol.md)

### Access across model families

|Host|MIGRO LoCoMo composite|Questions|
|---|---:|---:|
|Qwen3-4B-Instruct-2507|54.13|1,986|
|Phi-3-mini-4k-instruct|49.77|1,986|
|Mistral-7B-Instruct-v0.3|45.69|1,986|

The receiving-host evaluations demonstrate access to an existing memory state through different host interfaces. Qwen uses the AA write trajectory; Phi and Mistral use the AB reference with their selected frozen interfaces. These trajectories and interface choices matter when comparing rows.

MIGRO, PlugMem and mem0 have completed **State Transfer** entries in the migration snapshot. Target-built controls are explicitly labeled **Full-History Rebuilding**.

[Labeled migration results](results/migration_snapshot.csv) · [Source trajectories and interface selection](docs/migration_protocol.md)

<details>
<summary><strong>Evaluation details and interpretation</strong></summary>

The LoCoMo composite was selected post hoc. It weights all 1,986 questions equally: 1,540 category 1–4 question scores and 446 category 5 scores. It is not official LoCoMo token F1. The host rows do not form a strict controlled ranking of model capability.

For MIGRO's AB reference, the existing state contains a Qwen-written prefix followed by Phi continuation. Phi uses the selected step-0 identity interface. Mistral uses a read interface with 131,072 parameters, calibrated on 64 unique public human-QA examples for 64 optimizer updates. The backbones and shared memory core are frozen; LoCoMo answers are not calibration targets.

PlugMem and mem0 reuse their Qwen-built source states for their State Transfer rows. Their source trajectories differ from MIGRO's AB reference. A migration-loss claim requires a matched target-backbone rebuilding reference.

LongMemEval-S and MemOps use the recorded auxiliary judge protocol. BEAM uses category-specific scoring with auxiliary judging where required. These are research-harness evaluations, not official leaderboard reproductions. Baselines retain their declared native read paths; inference compute is not equalized.

</details>

### Mechanisms under examination

The experiment contracts separate questions about the complete system from questions about individual components.

|Research question|Controlled comparison|Specification|
|---|---|---|
|Does the memory system provide useful evidence?|Full memory, BM25, recent history and no-memory controls, with target-native rebuilding as a reference.|[Core transfer](configs/ablations/core_transfer.json)|
|What does the host interface contribute?|Boundary composition crossed with frozen calibration.|[Transport and calibration](configs/ablations/transport_composition.json)|
|What is the address space–quality tradeoff?|Anchor + residual, anchor-only and original full addresses.|[Address representation](configs/ablations/addresses.json)|
|What does persistence coding save?|Full lossless selection, existing literal candidates and an uncompressed archive.|[Persistence encoding](configs/ablations/persistence.json)|

These configurations document the comparisons. Their presence is not a claim that every component has established an independent performance gain. Paired analysis operates on matching question IDs and preserves conversation clusters.

## Released components

The repository provides entry points into the implementation and the evaluation workflow.

|Component|Role|Source|
|---|---|---|
|**Transport**|Vocabulary-aware piecewise transport, boundary composition and lexical mapping.|[`migro.transport`](src/migro/transport.py)|
|**Calibration**|Residual and evidence-conditioned host interfaces.|[`migro.calibration`](src/migro/calibration.py)|
|**Reference writing**|Frozen-host causal writing from observed history.|[`migro.native_writer`](src/migro/native_writer.py)|
|**Observation handling**|Deterministic formatting and temporal notes.|[`migro.observations`](src/migro/observations.py)|
|**Memory controls**|Portable BM25 and recent-window baselines.|[`migro.controls`](src/migro/controls.py)|
|**Paired analysis**|Matched-cohort comparison and conversation-level bootstrap intervals.|[`migro_eval.ablations`](src/migro_eval/ablations.py)|

### Installation

Use Python 3.10 or later. Reporting and the portable text controls use the standard library. The optional transport and analysis extras install the dependencies needed by those modules.

```bash
python -m pip install -e .
python -m pip install -e '.[transport,analysis]'
```

Transport and reference writing require application-supplied models and state interfaces. [Component contracts and usage](docs/implementation.md)

### Reporting and paired analysis

```bash
# Render the checked-in aggregate results.
python scripts/render_results.py

# Aggregate an already-scored LoCoMo cohort.
python scripts/summarize_locomo.py /path/to/scored_records.json --output summary.json

# Inspect the paired-ablation analysis interface.
python scripts/summarize_ablations.py --help

# Run the component and reporting checks.
python -m unittest discover -s tests -v
```

The reporting tools consume scored records and require complete cohorts by default. Per-question scoring remains upstream. Optional-dependency tests skip when their dependencies are unavailable.

[Reporting input format](docs/reporting_tools.md) · [Paired analysis usage](docs/implementation.md#paired-analysis)

### Repository structure

```text
assets/                 Visual identity and system figures
src/migro/              Transport, calibration, writing and controls
src/migro_eval/         Scored-record aggregation and paired analysis
configs/ablations/      Component-level experiment contracts
docs/                   Architecture, protocols and implementation notes
results/                Dated aggregate results and evaluation metadata
scripts/                Reporting and analysis entry points
tests/                  Component and reporting checks
```

## Release status

This repository is being prepared for open-source release. The current upload contains selected implementation modules, experimental contracts and reporting tools. The full compressed-state engine, pretrained core/interface checkpoints, complete benchmark inference runners and datasets are outside this upload.

The numerical snapshot dates to November 1, 2026. The checked-in evaluation status is a dated record; pending runs have no assigned final score. Larger-read-budget candidates and storage-replica QA scores do not replace the frozen performance table.

Public licensing and full reproduction instructions will be specified for the open-source release. Third-party methods and datasets retain their respective licenses.

[Implementation scope](docs/implementation.md) · [Release scope](docs/release_scope.md) · [Third-party references](docs/third_party.md)
