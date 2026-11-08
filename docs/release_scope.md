# Release scope

Included code: shared-anchor memory and residual addresses; exact source-token bindings and metadata; snapshots, compaction and continued writing; C++ coding kernels and build command; host interfaces and readers; MAB preparation, incremental generation, matched read budgeting and format-aware scoring; MIGRO continued-writing generation with transfer-only and full-history rebuilding controls; reporting, paired analysis and targeted tests.

The [runtime asset release](https://github.com/zOUMINGXI/MIGRO/releases/tag/runtime-assets-20261108) adds the shared core, coding parameters, tokenizer/config files and frozen Phi/Mistral interface checkpoints. See [runtime setup](runtime.md). Backbone and reranker models, datasets and third-party packages are obtained separately. The release follows repository access: it is not an anonymous public download while this repository is private.

The [November 8 aggregate update](../results/latest/README.md) includes complete primary benchmark scores, Qwen scale-up, MAB budget comparisons, heterogeneous host results, continued writing and storage measurements. Pending runs have no final score. Historical November 1 tables remain archived. State Transfer and Full-History Rebuilding retain their per-experiment meanings.

Complete baseline adapters, remaining older benchmark launchers and the six-GPU cluster scheduler remain outside this upload. This is not yet a self-contained full-table reproduction bundle. The shared asset download does not alter licensing of third-party models or data.

[Runtime assets](runtime.md) · [Experiment commands](experiments.md) · [Component contracts](implementation.md)
