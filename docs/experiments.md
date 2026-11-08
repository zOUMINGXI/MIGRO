# Experiment entry points

Set up [runtime assets](runtime.md) first. Commands run MIGRO; third-party baseline adapters remain outside this staged release.

## MemoryAgentBench

Provide an upstream MemoryAgentBench checkout and its released Parquet files. The research snapshot used upstream commit 45ac4bf400d42ddc35d9d824300d490b9145c6f3. Install that checkout's evaluation dependencies too.

~~~bash
migro-mab-prepare --upstream-root /path/to/MemoryAgentBench --data-dir /path/to/parquet --work-dir runs/mab
migro-mab-units --work-dir runs/mab
migro-mab-run --work-dir runs/mab --output runs/mab/migro2000 --source-token-cap 2000
migro-mab-score --work-dir runs/mab --upstream-root /path/to/MemoryAgentBench --output runs/mab/migro2000 --method migro
~~~

Preparation selects fourteen recorded task profiles and separates observations, questions and labels. Original question IDs are retained for the LongMemEval abstention rubric. Oversized official chunks are split without dropping text. Internal memory units target 448 source-token body spans and must fit 512 formatted source tokens; original turn order and IDs remain linked.

Generation opens memory units first, writes chronologically with native window 4 and checkpoints every 32 units. Final questions open after the history is complete; labels are not loaded by writer or generator. Retrieval intent removes demonstrations only for the two recorded long-range profiles; the final host prompt stays intact. Reranker query cap is 128; pair cap is 384. Output limits come from each upstream task profile.

Use --source-token-cap 8000 with a separate output directory for the larger evidence budget. Split independent histories across GPUs with --shards N --shard-index I and separate output directories. Concatenate completed shard JSONL files into the chosen scoring output's predictions.jsonl. Scoring rejects missing or duplicate IDs and mixed read budgets. Run settings are saved with each output directory; use a new directory when changing the budget, shard, input manifest or receiving host. Interrupted checkpoints retain the last complete snapshot.

matched_read_budget.py wraps TransMem's native suffix planner. It counts the raw suffix with MIGRO's source tokenizer while retaining native overflow state. This equalizes source-token evidence, not all computation or model state.

Scoring retains strict outputs and exports RESULTS_FORMAT_CORRECTED.json: complete ICL numeric labels and valid DetectiveQA JSON.answer extraction, followed by unchanged upstream task metrics. LongMemEval and summary tasks use the disclosed auxiliary judge, default gpt-5.4-mini. Set OPENAI_API_KEY and optionally OPENAI_BASE_URL; --judge-model, --base-url and --api-key-env are configurable. Completed judgments are reused only for the selected judge model. No account profile is embedded. The four-ability macro is custom, not an official MAB total. Provide upstream's Recsys entity2id.json at runs/mab/assets/entity2id.json if absent from its processed-data directory.

## Continued writing after handoff

Provide a JSON array with one entry per chronological history:

~~~json
[
  {
    "case_id": "conversation_1",
    "observations": "conversation_1/observations.json",
    "cut": 50,
    "questions": "conversation_1/questions.json"
  }
]
~~~

Paths are relative to the manifest. Observations follow the runtime schema. Questions contain id, case_id, benchmark: "locomo", system, messages and retrieval_query. Keep answer labels and grouping annotations in separate scoring files.

~~~bash
migro-continue --manifest data/continuation.json --output runs/continuation --phase source
migro-continue --manifest data/continuation.json --output runs/continuation --phase target --target-host C
~~~

Host A is Qwen4B, B is Phi and C is Mistral. Source writes the prefix. Target completes all states before opening any final questions:

- transfer_only: receive prefix state and append nothing.
- transfer_continue: receive that same state and write only the receiver's chronological suffix.
- target_rebuild: reconstruct the full history on the receiver from empty state.

The native writer uses window 4 and batches of 8, matching the research continuation path. Prior token bindings and metadata are checked after append. Every arm receives the same final-history question cohort and is scored against complete-history labels, including transfer-only. Old/new/cross groups require separate annotations; revision claims require actual revision labels.

After applying the recorded question scorer, scripts/summarize_ablations.py compares matching scored arms with conversation-level resampling. This upload does not include a new LoCoMo raw-answer scorer or all baseline launchers. The command preserves MIGRO's construction protocol and emits predictions; its presence does not assert reproduction of a table score.
