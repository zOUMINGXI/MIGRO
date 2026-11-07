# The Handoff

[**Open the interactive demo ↗**](https://migro-handoff.lucentsole3.chatgpt.site) — works directly in your browser, with no download or installation. Switch cases, compare recorded model answers and inspect the source evidence. The online demo is public; the research repository remains private.

For offline playback, [download MIGRO-Handoff.zip](https://github.com/zOUMINGXI/MIGRO/raw/refs/heads/main/demo/MIGRO-Handoff.zip), unzip it and open **MIGRO-Handoff.html**. The handwritten interface, cases, source correspondence and recorded answers are bundled for offline use.

![Compare project decisions before and after a model handoff](preview.webp)

## The kitchen refit

Leah is living in her flat during a kitchen renovation. Her building manager, family, trades and supplier leave nine messages in April. Her assistant changes hosts. In May, a cabinet quote is replaced, delivery moves, a drawing arrives for sign-off and the plumber wants a confirmed date.

The notebook opens at the drawing sign-off. Choose any of the six decisions in the inbox. Compare **Transfer + append** with **Only later messages**, **Transfer only** or **Full history rebuild** using the same question. For example, quote B supplies the new subtotal and delivery fee; the hard spending limit and deposit are in earlier correspondence. The delivery question requires the supplier’s revised options, the lift rules and Noor’s availability. For plumbing, a proposed template date is distinct from a confirmed worktop fitting.

**Show retrieved notes** displays the slots returned to each model. **Check this decision against the complete record** opens an authored checklist with links to the dated messages. **Show decision references** filters the ledger to those messages. The checklist is separate from the generator and is only used for reading the demo.

This is a fictional household project. The dates, speakers and documents are written for this demonstration. The responses shown in all five conditions are the actual model outputs.

## Shipping the MIGRO demo

A second case reconstructs a real maintainer problem from this repository. A collaborator can replay the offline demo, wants to change its inputs and has only a fresh clone. Answering requires both the new download instructions and the earlier model/asset setup. The project records are paraphrases of repository materials and the support questions are scripted. Source records: [runtime publication](https://github.com/zOUMINGXI/MIGRO/commit/d1f84460d286a21c66a9bdc7aec2bc22ab6d2a04), [demo entry point](https://github.com/zOUMINGXI/MIGRO/commit/55ba0d0aa7ccc218287aed33fa60fa055f08a077), [ZIP fix](https://github.com/zOUMINGXI/MIGRO/commit/ed9752d28943cbcfacee05fd03d3ae83bfef72f2), [preview update](https://github.com/zOUMINGXI/MIGRO/commit/ca8a267e4cc667cff14db4f9c076192bedd5c2db).

The two earlier field-recording sessions are also available from **Open a case**.

## How to read the comparison

| Condition | History available to the host | Receiver writes in the kitchen case |
|---|---|---:|
| Qwen · source | April messages under the source host | 0 |
| Transfer only | Mistral receives Qwen’s saved state | 0 |
| Transfer + append | Mistral receives that state and writes the May messages | 6 |
| Only later messages | Mistral starts empty and writes only the May messages | 6 |
| Full history rebuild | Mistral writes the complete correspondence from empty memory | 15 |

Within each project case the system instruction, questions and generation limit are shared across conditions. The project cases use MIGRO's native state, writer, host transport and evidence reader, with an assistant-style request and a 256-token answer budget. The older short examples keep their original benchmark-style prompt. Counts describe observations written, and bytes refer to the serialized private snapshot. These individual case studies have no assigned benchmark score.

## Run your own session

The online demo replays bundled model runs. Its **Edit case** action exports your edited session; generating fresh answers requires the local runtime below.

Configure the [runtime assets](../docs/runtime.md), including the frozen shared core, coding kernels, tokenizers, Qwen3-4B-Instruct-2507, Mistral-7B-Instruct-v0.3, the Mistral interface and the evidence reranker.

```sh
python demo/serve.py
```

Open `http://127.0.0.1:18764`, select **Edit & rerun** and enter one message or question per line. A dated message uses `YYYY-MM-DD | Speaker | Channel | Message`; plain notes also work. The local service defaults to CPU. To run on a suitable GPU use `MIGRO_DEVICE=cuda python demo/serve.py`.

```sh
MIGRO_DEVICE=cpu python demo/run.py --case demo/renovation.json --output trace.json
python demo/serve.py --trace trace.json
```

The runner checks the handoff snapshot byte for byte, preserves source bindings and metadata across append, and checks read-only queries. Each rebuilding condition starts from an empty state. Edited sessions receive fresh generations in all conditions. Authored decision checks are attached only to their original case.

[Kitchen correspondence](renovation.json) · [Kitchen model outputs](renovation.recorded.json) · [Release correspondence](release.json) · [Release model outputs](release.recorded.json) · [Handwriting credits](../assets/lettering-credits.md)
