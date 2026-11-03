# Implementation components

This staged code upload adds selected implementation components. It is not the complete state engine or an end-to-end benchmark reproduction package. Existing numerical snapshots retain their original dates and values.

## Included

- `migro.transport`: valid-vocabulary piecewise transport, boundary-composed transport, lexicon construction and query-cache handling.
- `migro.calibration`: memory residual calibration and the evidence-conditioned assistant-boundary adapter; zero up-projection weights give an identity residual.
- `migro.native_writer`: frozen-host causal reference writing from observed history.
- `migro.observations`: deterministic observation formatting and temporal notes.
- `migro.controls`: portable BM25 and recent-window controls.
- `migro_eval.ablations`: paired comparisons of already-scored conditions, with whole-conversation bootstrap intervals.

The writer requires an application-supplied state exposing `core.codec.tokenizer`, `meta`, `mem.read_slot` and `append_many`. Its bridge supplies a frozen decoder-only host exposing `host.model.layers`, tokenizer, device and context limit. Callers freeze the host before use. The compressed shared-anchor state backend, shared-core weights, calibration checkpoints and benchmark inference harness are not bundled.

## Install and check

```bash
python -m pip install -e .
python -m unittest discover -s tests -v
python -m pip install -e '.[analysis,transport]'
```

Core reporting and BM25/recency controls use the standard library. Analysis requires NumPy. Transport, calibration and writing require PyTorch; composed transport also uses NumPy and Hugging Face tokenizers. Tests skip unavailable optional dependencies.

## Portable control example

```python
import json
from migro.controls import build_bm25_state, select_bm25, packet_source_ids
# Illustrative IDs only; supply real source-tokenizer IDs for real histories.
observed = [{"slot": 0, "text": "Ada visited Kyoto", "source_ids": [10, 11]},
            {"slot": 1, "text": "Ada works in London", "source_ids": [20, 21]}]
state = json.loads(json.dumps(build_bm25_state(observed)))
slots = select_bm25(state, "Where did Ada visit?", budget=2000)
ids = packet_source_ids(state, slots, separator_ids=[99])
```

Build indexes from observed history before exposing future QA. Text is used during index construction; exact source IDs and supplied metadata persist. Recent-window construction retains a contiguous suffix of whole observations. It does not skip an oversized latest observation to retrieve older facts. BM25 retains all supplied history, fits complete ranked rows into the source-token budget and delivers them chronologically.

## Paired analysis

Supply a JSON array per condition for one backbone. Each already-scored row has `id`, `case_id`, and `score` in `[0,1]`; `host` or `backbone` may be supplied. Per-question answer scoring happens upstream.

```bash
python scripts/summarize_ablations.py \
  --condition full=/path/to/full_scored.json \
  --condition bm25=/path/to/bm25_scored.json \
  --reference full --output /path/to/comparison.json
```

Defaults require 1,986 matching questions per condition. The LoCoMo cohort has 10 conversation clusters. Resampling preserves whole conversations and question weighting. Intervals are exploratory and unadjusted; crossing zero does not establish equivalence. The composite is not official LoCoMo token F1.

## Experiment contracts

`configs/ablations/` records predefined groups: core efficacy versus BM25/recency/no memory, with target-native rebuilding as a positive reference; boundary composition crossed with frozen calibration; address representation and its space-quality tradeoff; and exact persistence encoding. These contracts do not replace the omitted backend.

`State Transfer` and `Full-History Rebuilding` label actual run operations. Whole-memory gains do not identify one submodule. Native-text reading is a compatibility/fidelity reference: lexical transport often approximates native retokenization. All predefined results are retained regardless of which condition scores higher.

## Assets and attribution

Supply models, tokenizers, checkpoints and benchmark data separately under their original licenses. This upload contains no benchmark observations, labels, predictions, model artifacts, machine addresses or operational credentials. Attribution remains in `CONTRIBUTORS.md`.
