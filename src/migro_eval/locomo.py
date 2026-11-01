"""Aggregate scored LoCoMo questions without redefining their per-question scorer."""
from __future__ import annotations
import math
from collections import defaultdict
from collections.abc import Iterable, Mapping
from typing import Any
MAIN_QUESTIONS = 1540
CATEGORY5_QUESTIONS = 446
TOTAL_QUESTIONS = MAIN_QUESTIONS + CATEGORY5_QUESTIONS
def _number(value: Any, upper: float) -> float:
    if isinstance(value, bool):
        raise ValueError("Scores must be numeric, not booleans")
    number = float(value)
    if not math.isfinite(number) or not 0 <= number <= upper:
        raise ValueError(f"Score must be finite and within [0, {upper}]")
    return number
def composite_from_components(main_f1: float, category5_score: float) -> float:
    """Return the custom all-question score on a 0–100 scale.

    The two input components must already be on a 0–100 scale. This reporting
    metric was selected post hoc; it is not the official LoCoMo token F1.
    """
    main = _number(main_f1, 100)
    category5 = _number(category5_score, 100)
    return (MAIN_QUESTIONS * main + CATEGORY5_QUESTIONS * category5) / TOTAL_QUESTIONS
def summarize_scored(
    records: Iterable[Mapping[str, Any]], *, require_full: bool = True
) -> list[dict[str, Any]]:
    """Aggregate per-question scores in [0, 1], separately per method/backbone.

    Required keys: id, method, category, score, plus backbone or host.
    Benchmark inference and category-specific scoring happen upstream. Full
    results require 1,540 category 1–4 and 446 category 5 unique questions.
    """
    groups: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        method = str(record["method"])
        backbone = record.get("backbone", record.get("host"))
        if not method or backbone is None:
            raise ValueError("Each question requires a method and backbone/host")
        category = record["category"]
        if isinstance(category, bool) or category not in (1, 2, 3, 4, 5):
            raise ValueError("Category must be an integer from 1 through 5")
        _number(record["score"], 1)
        groups[(method, str(backbone))].append(record)
    if not groups:
        raise ValueError("No scored questions")
    output = []
    for (method, backbone), rows in sorted(groups.items()):
        ids = [str(row["id"]) for row in rows]
        if len(set(ids)) != len(ids):
            raise ValueError(f"Duplicate question IDs for {method}/{backbone}")
        category5_n = sum(row["category"] == 5 for row in rows)
        main_n = len(rows) - category5_n
        complete = (main_n, category5_n) == (MAIN_QUESTIONS, CATEGORY5_QUESTIONS)
        if require_full and not complete:
            raise ValueError(
                f"Incomplete cohort for {method}/{backbone}: "
                f"{main_n} main and {category5_n} category-5 questions"
            )
        output.append(dict(method=method, backbone=backbone,
            metric="custom_all_question_composite",
            score=100 * math.fsum(float(row["score"]) for row in rows) / len(rows),
            questions=len(rows), complete=complete, official_f1=False))
    return output
