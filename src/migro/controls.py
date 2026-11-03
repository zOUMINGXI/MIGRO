"""Portable lexical and recent-window controls; build only from observed history."""
from __future__ import annotations
import copy
import math
import re
from collections import Counter, defaultdict

def words(text):
    return re.findall(r"[a-z0-9]+", text.lower())

def build_bm25_state(observations, *, k1=1.2, b=0.75):
    """Chronological rows require slot, text and exact source_ids."""
    rows, lengths, postings = [], [], defaultdict(list)
    for position, item in enumerate(observations):
        counts = Counter(words(item["text"]))
        lengths.append(sum(counts.values()))
        for term, count in counts.items():
            postings[term].append([position, count])
        rows.append(copy.deepcopy({k: v for k, v in item.items() if k != "text"}))
    if not rows:
        raise ValueError("A memory state requires observations")
    if len({r["slot"] for r in rows}) != len(rows):
        raise ValueError("Observation slots must be unique")
    n = len(rows)
    return dict(schema="migro-bm25-control", rows=rows, index=dict(
        postings=dict(postings), lengths=lengths, avgdl=sum(lengths) / n, k1=k1, b=b,
        idf={term: math.log(1 + (n - len(values) + .5) / (len(values) + .5))
             for term, values in postings.items()}))

def select_bm25(state, query, *, budget=2000, separator_tokens=1):
    """Rank complete observations; zero-score ties favor recent rows."""
    if budget < 0 or separator_tokens < 0:
        raise ValueError("Token budgets cannot be negative")
    rows, ix = state["rows"], state["index"]
    scores = [0.0] * len(rows)
    k1, b = ix["k1"], ix["b"]
    for term in set(words(query)):
        for i, tf in ix["postings"].get(term, []):
            norm = 1 - b + b * ix["lengths"][i] / max(ix["avgdl"], 1e-9)
            scores[i] += ix["idf"][term] * tf * (k1 + 1) / (tf + k1 * norm)
    order = sorted(range(len(rows)), key=lambda i: (-scores[i], -i))
    chosen, used = [], 0
    for i in order:
        cost = len(rows[i]["source_ids"]) + separator_tokens
        if used + cost <= budget:
            chosen.append(i)
            used += cost
    return [rows[i]["slot"] for i in sorted(chosen)]

def build_recent_state(observations, *, budget=2000, separator_tokens=1):
    """Retain a contiguous suffix of complete observations, without queries."""
    if budget < 0 or separator_tokens < 0:
        raise ValueError("Token budgets cannot be negative")
    selected, used = [], 0
    for row in reversed(list(observations)):
        cost = len(row["source_ids"]) + separator_tokens
        if used + cost > budget:
            break
        selected.append(copy.deepcopy({k: v for k, v in row.items() if k != "text"}))
        used += cost
    selected.reverse()
    return dict(schema="migro-recent-control", rows=selected, source_token_budget=budget)

def packet_source_ids(state, slots, *, separator_ids=()):
    """Recover exact selected IDs with explicit observation separators."""
    rows = {r["slot"]: r for r in state["rows"]}
    return [token for slot in slots for token in
            list(rows[slot]["source_ids"]) + list(separator_ids)]
