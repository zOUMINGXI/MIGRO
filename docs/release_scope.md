# Release scope

Included code: shared-anchor memory and residual addresses; exact source-token bindings and metadata; snapshots, compaction and continued writing; C++ coding kernels and build command; host interfaces and readers; MAB preparation, incremental generation, matched read budgeting and format-aware scoring; MIGRO continued-writing generation with transfer-only and full-history rebuilding controls; reporting, paired analysis and targeted tests.

The [shared runtime bundle](https://github.com/zOUMINGXI/MIGRO/releases/tag/runtime-assets-20261112) includes component provenance, license notices and deployment-only checkpoint metadata. See [runtime setup](runtime.md). Backbone and reranker models, datasets and third-party packages are obtained from their original sources.

The [selected result summaries](../results/README.md) include primary benchmarks, same-host scale-up, migration and matched-state storage. README tables follow the current manuscript. Detailed experimental data, intermediate artifacts and run-status exports are reserved for a later release.

Complete baseline adapters, remaining older benchmark launchers and the six-GPU cluster scheduler remain outside this upload. This is not yet a self-contained full-table reproduction bundle. The shared asset download does not alter licensing of third-party models or data.

[Runtime assets](runtime.md) · [Experiment commands](experiments.md) · [Component contracts](implementation.md)
