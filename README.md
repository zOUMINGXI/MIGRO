<p align="center">
  <strong>English</strong> &nbsp; · &nbsp; <a href="README.zh-CN.md">简体中文</a> &nbsp; · &nbsp; <a href="README.es.md">Español</a>
</p>

<p align="center">
  <img src="assets/migro-launch.svg" alt="MIGRO — Models evolve. Memory endures. Persistent memory across language models." width="1600">
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2611.15315"><img src="https://img.shields.io/badge/arXiv-2611.15315-B75125?style=flat-square" alt="Paper on arXiv"></a>
  <a href="https://zoumingxi.github.io/MIGRO/"><img src="https://img.shields.io/badge/Demo-Try%20online-E87524?style=flat-square" alt="Interactive demo"></a>
  <img src="https://img.shields.io/badge/Python-3.10%2B-315F78?style=flat-square&amp;logo=python&amp;logoColor=white" alt="Python 3.10 and later">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache--2.0-315F78?style=flat-square" alt="Apache-2.0"></a>
</p>

<p align="center">
  <strong>Persistent memory across language models</strong><br>
  A shared substrate for accumulated experience. Adaptable interfaces for the models that use it.
</p>

<p align="center">
  <a href="#quick-start">Quick start</a> &nbsp; · &nbsp;
  <a href="https://arxiv.org/abs/2611.15315"><strong>Paper ↗</strong></a> &nbsp; · &nbsp;
  <a href="https://zoumingxi.github.io/MIGRO/"><strong>Try the demo ↗</strong></a> &nbsp; · &nbsp;
  <a href="#why-migro">Overview</a> &nbsp; · &nbsp;
  <a href="#system-design">Architecture</a> &nbsp; · &nbsp;
  <a href="#experimental-evidence">Evidence</a> &nbsp; · &nbsp;
  <a href="docs/README.md">Documentation</a> &nbsp; · &nbsp;
  <a href="#citation">Citation</a>
</p>

---

## Why MIGRO

**MIGRO keeps long-term experience available as language models change.** It stores observations, historical relationships and retrieval addresses in a persistent state that successive hosts can read and extend.

- **Useful memory.** Exact evidence and query-time selection support answers about events, revisions and context across long histories.
- **Portable experience.** Host-specific interfaces let a receiving model access inherited state and keep writing after handoff.
- **Economical growth.** Compact semantic addresses and reversible archives reduce persistent storage while retaining the saved evidence.

## News

- **2026-11-12** — [Shared runtime assets released](https://github.com/zOUMINGXI/MIGRO/releases/tag/runtime-assets-20261112), with core and host-interface checkpoints, setup and verification tools.
- **2026-11-11** — [MIGRO technical report published on arXiv](https://arxiv.org/abs/2611.15315).

## Quick start

**Try it in your browser.** [Open the interactive demo](https://zoumingxi.github.io/MIGRO/) to compare recorded model answers and inspect retrieved evidence. [Download the offline demo](https://github.com/zOUMINGXI/MIGRO/raw/refs/heads/main/demo/MIGRO-Handoff.zip) for local replay.

**Install from source.** The base installation provides reporting tools and portable text controls.

~~~bash
git clone https://github.com/zOUMINGXI/MIGRO.git
cd MIGRO
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python scripts/render_results.py --help
~~~

**Run the memory system.** Follow [runtime setup](docs/runtime.md) to install runtime dependencies, download shared assets, compile the coding kernels and run the provided verification. Then use [the local demo guide](demo/README.md#run-your-own-session) to generate fresh answers with the required backbone and reranker models.

[Documentation index](docs/README.md) · [Experiment commands](docs/experiments.md) · [Release scope](docs/release_scope.md)

## System design

<p align="center">
  <img src="assets/migro-system.svg" alt="MIGRO reference architecture: observed experience enters a source-host writer; persistent memory stores evidence, semantic addresses and encoded state; a transport and calibration interface connects it to receiving hosts." width="1600">
</p>

### The Handoff · interactive demo

A kitchen supplier asks for sign-off just as Leah changes assistants. The new quote, drawing and delivery options are in the latest emails; her spending cap, site measurements and building restrictions are in April’s correspondence. Take over the decision desk: make your own call, recover the earlier context and reconsider. Then compare the actual answers from five memory conditions. A second case reconstructs our own demo-download support issue from the release commits.

[![The Handoff — recorded Qwen to Mistral session](demo/preview.webp)](https://zoumingxi.github.io/MIGRO/)

[**Open the interactive demo ↗**](https://zoumingxi.github.io/MIGRO/) · [Download the offline demo (.zip)](https://github.com/zOUMINGXI/MIGRO/raw/refs/heads/main/demo/MIGRO-Handoff.zip) · [Run your own session](demo/README.md#run-your-own-session)

Open the online demo in any browser—no download or installation required. Explore the recorded model runs, switch memory conditions and inspect their retrieved evidence. To generate fresh answers from your own inputs, use the local runtime linked above.

### A persistent substrate with adaptable host connections

MIGRO separates the stored experience from the machinery used to expose it to a particular host.

**Writing preserves a connection to observed history.** Native reference writing uses the source host to construct memory from observations. Persistent payloads and metadata retain the relationship between stored content and its originating experience.

**Addressing and encoding serve different roles.** Semantic addresses support evidence selection. The anchor-and-residual representation is evaluated against anchor-only and full-address controls. Persistence encoding is evaluated separately, including exact restoration of source IDs, address geometry and metadata.

**The read interface handles the receiving host.** Lexical transport connects source representations to the target vocabulary and embedding interface. Calibration adapts the host connection where the protocol permits it. This factorization makes host-specific adaptation an explicit part of the system.

The diagram describes the research architecture. The code release includes the persistent state backend, exact coding kernels and benchmark entry points. Shared checkpoints and datasets are supplied separately; see the [runtime setup](docs/runtime.md).

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

The report evaluates answer quality, cross-host continuity and persistent storage. These measurements use their respective cohorts and accounting scopes.

| Study | Representative result |
|---|---|
| Long-term QA · LongMemEval-S | **68.40** under the Qwen 4B reference host |
| Continued writing · Qwen 4B → Mistral 7B | **45.13**, compared with **45.19** for target-side full-history rebuilding |
| Million-token BEAM histories | **1.857 MB** private state per history, averaged over five histories |
| Shared deployment · 1,000 independent memories | **2.154 MB per memory**, including required shared assets |

[All benchmark and baseline tables](docs/evaluation.md) · [Metrics and scoring](docs/performance_protocol.md) · [Migration settings](docs/migration_protocol.md)

<p align="center">
  <a href="assets/joint_frontier.png"><img src="assets/joint_frontier.png" alt="Joint memory frontier: continued writing after Qwen 4B to Mistral 7B handoff, comparing answer quality, private memory footprint and rebuilding quality retained." width="1600"></a>
</p>

<p align="center">
  <strong>Carry the experience. Keep writing.</strong><br>
  Continued writing after a Qwen 4B → Mistral 7B handoff: answer quality, private memory footprint and quality retained relative to target-side rebuilding.
</p>

## Released components

The repository provides entry points into the implementation and the evaluation workflow.

|Component|Role|Source|
|---|---|---|
|**Persistent state**|Shared anchors and residuals, exact content bindings, save/restore and continued writes.|[`migro.memory`](src/migro/memory.py)|
|**Exact coding**|Literal and causal content candidates, address/archive coding and native kernels.|[`migro.backend`](src/migro/backend)|
|**Benchmark runs**|Incremental MAB, matched read budgets, format-aware scoring and post-transfer continuation.|[Running experiments](docs/experiments.md)|
|**Transport**|Vocabulary-aware piecewise transport, boundary composition and lexical mapping.|[`migro.transport`](src/migro/transport.py)|
|**Calibration**|Residual and evidence-conditioned host interfaces.|[`migro.calibration`](src/migro/calibration.py)|
|**Reference writing**|Frozen-host causal writing from observed history.|[`migro.native_writer`](src/migro/native_writer.py)|
|**Observation handling**|Deterministic formatting and temporal notes.|[`migro.observations`](src/migro/observations.py)|
|**Memory controls**|Portable BM25 and recent-window baselines.|[`migro.controls`](src/migro/controls.py)|
|**Paired analysis**|Matched-cohort comparison and conversation-level bootstrap intervals.|[`migro_eval.ablations`](src/migro_eval/ablations.py)|

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
results/                Selected manuscript summary tables (CSV / JSON)
scripts/                Reporting and analysis entry points
tests/                  Component and reporting checks
```

## Release status

The [shared runtime bundle](https://github.com/zOUMINGXI/MIGRO/releases/tag/runtime-assets-20261112) and implementation are available. The release includes persistent memory, coding kernels, host interfaces, setup and verification helpers, the demo and documented benchmark entry points. Backbone and reranker models and datasets are obtained from their publishers. The [release scope](docs/release_scope.md) lists available components and remaining reproduction work.

MIGRO-owned code and runtime assets use [Apache-2.0](LICENSE). Bundled upstream components retain the licenses described in [third-party notices](THIRD_PARTY_NOTICES.md).

## Contributing

Contributions to documentation, integrations, memory interfaces and evaluation tooling are welcome. Read the [contribution guide](CONTRIBUTING.md), [report a bug](https://github.com/zOUMINGXI/MIGRO/issues/new?template=bug_report.md), or [propose an improvement](https://github.com/zOUMINGXI/MIGRO/issues/new?template=feature_request.md).

## Citation

If MIGRO is useful in your research, please cite the [technical report](https://arxiv.org/abs/2611.15315). Machine-readable metadata is available in [CITATION.cff](CITATION.cff).

~~~bibtex
@article{zou2026migro,
  title = {{MIGRO}: A Decoupled and Portable Long-Term Memory System for Language Models},
  author = {Zou, Mingxi and
            Xu, Yinghui and
            Chen, Jiaxiang and
            Liang, Langzhang and
            Zhu, Wei and
            Ye, Peng and
            Zhao, Runkai and
            Jiang, Ruoxi and
            Chen, Sijin and
            Wang, Song and
            Wang, Zhuo and
            Qu, Lizhen and
            Xu, Zenglin},
  year = {2026},
  eprint = {2611.15315},
  archivePrefix = {arXiv},
  primaryClass = {cs.AI},
  url = {https://arxiv.org/abs/2611.15315}
}
~~~
