# The Handoff

[**Download MIGRO-Handoff.zip**](https://github.com/zOUMINGXI/MIGRO/raw/refs/heads/main/demo/MIGRO-Handoff.zip), unzip it and open **MIGRO-Handoff.html**. Both cases, the handwriting font and the recorded answers are inside the offline page.

![Explore model handoff, continued writing and full rebuilding](preview.webp)

## Explore the field notebook

Open **The midnight field session**. Qwen writes four observations, including the location of a hydrophone and the required audio format. Mistral receives that state and writes three more notes about a changed venue, a collection time and transport. **The listening session** is a shorter two-note example.

Use **Write**, **Hand off** and **Continue** to follow the same question across time. The comparison menu shows the verbatim response to that question under another condition.

| Condition | What the host reads | Receiver writes in the field case |
|---|---|---:|
| Qwen · source | Source history written by Qwen | 0 |
| Transfer only | Mistral opens Qwen’s state | 0 |
| Transfer + append | Mistral opens that state and adds the later notes | 3 |
| New notes only | Mistral starts empty and writes the later notes | 3 |
| Full history rebuild | Mistral writes the entire history from scratch | 7 |

Try **Delivery brief**. The format is in old note N2, while the time is in new note N6. Switch to **Full history rebuild** to compare answers and receiver write counts. **Show retrieved notes** lists everything returned by the retriever for each answer. The source/update ledger below contains the exact text. Stored bytes count the serialized state and exclude shared assets.

These are two fictional demonstrations. Inspect all questions, including the one with an unknown detail. Each condition uses the same question text. The answers come from the models. The wire animation is an illustration.

## Run your own session

Configure the [runtime assets](../docs/runtime.md) first, including the frozen shared core, coding kernels, tokenizers, Qwen3-4B-Instruct-2507, Mistral-7B-Instruct-v0.3, the Mistral interface and the evidence reranker. Live generation requires these assets. The offline page works immediately.

From the repository root:

```sh
python demo/serve.py
```

Open `http://127.0.0.1:18764`, select **Edit & rerun**, and enter one observation or question per line. The server generates every comparison again with both models. It uses CPU by default. On a machine with sufficient GPU memory use `MIGRO_DEVICE=cuda python demo/serve.py`.

For a command-line run:

```sh
MIGRO_DEVICE=cpu python demo/run.py --case demo/session.json --output trace.json
python demo/serve.py --trace trace.json
```

The runner checks that the transferred bytes restore exactly, that appending preserves the earlier token bindings and metadata, and that question answering leaves the state unchanged. Both target-side rebuilding conditions start from empty memory. Shown write counts are observations ingested, not time estimates.

[Field case and model outputs](session.recorded.json) · [Short case and model outputs](starter.recorded.json) · [Handwriting credits](../assets/lettering-credits.md)

The interface embeds Kalam Regular and Bold from Google Fonts. The SIL Open Font License and attribution are included inside the standalone HTML.
