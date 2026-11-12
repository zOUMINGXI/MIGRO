<p align="center">
  <strong>English</strong> &nbsp; · &nbsp; <a href="runtime.zh-CN.md">简体中文</a> &nbsp; · &nbsp; <a href="runtime.es.md">Español</a>
</p>

# Run MIGRO

The shared runtime bundle includes the frozen core, coding tables, tokenizers and
Phi/Mistral read interfaces. It contains component provenance and license notices.
MIGRO-owned code and assets use Apache-2.0; bundled upstream components retain their
original licenses.

## Install

The exercised setup uses Linux, Python 3.10, a C++17 compiler and PyTorch 2.5.1.
Install a PyTorch build suitable for your hardware. The CUDA validation used:

~~~bash
git clone https://github.com/zOUMINGXI/MIGRO.git
cd MIGRO
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -e '.[runtime,benchmarks]' -c constraints/runtime-tested.txt
~~~

For CPU memory operations, install the CPU PyTorch wheel using
`https://download.pytorch.org/whl/cpu` in place of the CUDA index above.

## Download and verify the shared assets

~~~bash
export MIGRO_ASSETS="$PWD/migro-assets"
python scripts/setup_runtime.py --assets-dir "$MIGRO_ASSETS"
python scripts/verify_runtime.py --assets-dir "$MIGRO_ASSETS"
~~~

The setup command downloads the public bundle and compiles the three C++ coding
kernels. The verification command loads the actual core and both interface
checkpoints, writes observations, saves and reloads state, transfers it, appends
new observations, compacts it and restores the original bindings and metadata.

[Download the bundle](https://github.com/zOUMINGXI/MIGRO/releases/tag/runtime-assets-20261112)
· [Component sources and licenses](../THIRD_PARTY_NOTICES.md)
· [Provenance](../PROVENANCE.json)

The manifest in `ASSETS.json` lists the required payload files. The checkpoint export
preserves tensor values and dtypes and includes the resolved interface scope.
Local paths, training configurations and experiment bookkeeping are removed.

## Generate fresh answers

The generative demo requires external backbone and reranker models. Download them
from their publishers with the setup helper:

~~~bash
python scripts/setup_runtime.py --assets-dir "$MIGRO_ASSETS" --skip-assets \
  --models qwen mistral reranker
MIGRO_DEVICE=cuda python demo/run.py --case demo/starter.json --output trace.json
MIGRO_DEVICE=cuda python demo/serve.py --trace trace.json
~~~

Open `http://127.0.0.1:18764`. The captured session includes source-host answers,
transferred-state answers, continued writing, later-messages-only and full-history
rebuilding. Use the local page to edit a session and generate fresh answers.

Both backbones require several GB of model storage. The complete Qwen 4B → Mistral
7B demo was exercised on an H800 80GB. The public browser demo is also available for
recorded playback.

To use the Phi interface, add `--models phi reranker`. Model downloads use
`huggingface_hub` and its usual authentication settings when required by a publisher.

Existing model directories can be supplied before importing the runtime:

~~~bash
export MIGRO_QWEN_MODEL=/path/to/Qwen3-4B-Instruct-2507
export MIGRO_MISTRAL_MODEL=/path/to/Mistral-7B-Instruct-v0.3
export MIGRO_PHI_MODEL=/path/to/Phi-3-mini-4k-instruct
export MIGRO_RERANKER_MODEL=/path/to/ms-marco-MiniLM-L6-v2
~~~

## Python API

~~~python
from migro.memory import MemoryStore, load_core

core = load_core("cpu")
store = MemoryStore.create(core)
store.observe([{
    "speaker": "Ada", "role": "observed_human", "body": "I moved to Kyoto.",
    "session": "session_1", "date": "1 January, 2026", "dialogue_id": "s1:0"
}])
store.save("state.bin")
receiver = MemoryStore.load(core, "state.bin").handoff()
~~~

`MemoryRuntime.observe()` supplies native frozen-host writing and requires the
configured host model. Relative asset names preserve the saved-state format.
`MIGRO_NATIVE_DIR` relocates generated kernels and `MIGRO_PROFILES` selects another
reader-profile JSON.

## Checks

~~~bash
MIGRO_TEST_ASSETS=1 python -m unittest discover -s tests -v
python -m pip check
~~~

The bundle's `ASSETS.json`, `PROVENANCE.json`, `NOTICE` and license files are included
in the download. CPU state tests and real-model generation checks are summarized in
[the runtime validation record](../assets/runtime/validation.json).
