"""Paired analysis of already-scored conditions; no answer scorer is redefined."""
from __future__ import annotations
import math

def paired_summary(conditions, *, reference, expected_questions=1986, bootstrap=4000, seed=7):
    """Question-weighted effects with whole-conversation bootstrap intervals.

    Records require id, case_id and score in [0,1]. Optional host/backbone
    identifiers must describe one backbone. Intervals are exploratory,
    unadjusted intervals; crossing zero does not establish equivalence.
    Install migro-eval[analysis] for the optional NumPy dependency.
    """
    import numpy as np
    if bootstrap < 1:
        raise ValueError("bootstrap must be positive")
    if reference not in conditions:
        raise ValueError("Missing reference condition")
    indexed, hosts = {}, set()
    for name, records in conditions.items():
        rows = {}
        for record in records:
            qid, value = str(record["id"]), record["score"]
            if qid in rows:
                raise ValueError(f"Duplicate question in {name}: {qid}")
            if isinstance(value, bool) or not math.isfinite(float(value)) or not 0 <= float(value) <= 1:
                raise ValueError("Scores must be finite numbers in [0,1]")
            if not str(record["case_id"]):
                raise ValueError("A conversation ID is required")
            host = record.get("backbone", record.get("host"))
            if host is not None:
                hosts.add(str(host))
            rows[qid] = dict(case_id=str(record["case_id"]), score=float(value))
        if len(rows) != expected_questions:
            raise ValueError(f"Incomplete condition {name}: {len(rows)}/{expected_questions}")
        indexed[name] = rows
    if len(hosts) > 1:
        raise ValueError("Paired comparisons require the same backbone")
    baseline = indexed[reference]
    for name, rows in indexed.items():
        if set(rows) != set(baseline):
            raise ValueError(f"Question IDs differ in {name}")
        if any(rows[qid]["case_id"] != baseline[qid]["case_id"] for qid in rows):
            raise ValueError(f"Conversation assignment differs in {name}")
    cases = sorted({r["case_id"] for r in baseline.values()})
    if len(cases) < 2:
        raise ValueError("Cluster intervals require at least two conversations")
    rng = np.random.default_rng(seed)
    output = dict(reference=reference, questions=expected_questions, conversations=len(cases),
                  metric="custom_all_question_composite_0_100_not_official_F1",
                  scores={n: 100 * math.fsum(r["score"] for r in rr.values()) / len(rr)
                          for n, rr in indexed.items()}, paired={})
    for name, rows in indexed.items():
        if name == reference:
            continue
        buckets = {c: np.asarray([rows[qid]["score"] - r["score"] for qid, r in baseline.items()
                                 if r["case_id"] == c]) for c in cases}
        draws = [100 * np.concatenate([buckets[c] for c in rng.choice(cases, len(cases), replace=True)]).mean()
                 for _ in range(bootstrap)]
        output["paired"][name] = dict(
            delta_vs_reference=100 * float(np.concatenate(list(buckets.values())).mean()),
            conversation_cluster_95ci=np.percentile(draws, [2.5, 97.5]).tolist())
    return output
