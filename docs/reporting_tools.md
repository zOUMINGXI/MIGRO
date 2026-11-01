# Reporting tools

## Render the aggregate snapshot
From the repository root:

    python scripts/render_results.py

The command reads results/performance_snapshot.json and prints a Markdown table of complete entries.

## Aggregate scored LoCoMo records
Each input record contains:

    {"id":"question-1","method":"MIGRO","backbone":"example-model","category":1,"score":0.75}

Input scores must already be produced by the category-specific scorer and be within [0, 1]. Either backbone or host identifies the model group. Multiple method/model groups are aggregated independently. JSON arrays and JSONL are supported.

    python scripts/summarize_locomo.py scored.json --output summary.json

Full mode requires 1,540 category 1–4 records and 446 category 5 records, with unique question IDs per method/model. For explicit diagnostics only:

    python scripts/summarize_locomo.py partial.jsonl --allow-partial

Partial output is marked complete=false. Do not compare it as a final benchmark score. The tool does not generate model answers, infer missing labels or turn a partial cohort into a full one.

## Validate the aggregation
    python -m unittest discover -s tests -v

Tests cover question-level weighting, a frozen reference point, duplicate IDs, explicit partial results and invalid score scales.
