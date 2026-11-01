"""Render the checked-in aggregate performance snapshot as Markdown."""
from __future__ import annotations
import argparse, json
from pathlib import Path
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path(__file__).resolve().parents[1] / "results/performance_snapshot.json")
    args = parser.parse_args()
    data = json.loads(args.input.read_text())
    print("| Benchmark | Method | Backbone | Score | Questions |")
    print("|---|---|---|---:|---:|")
    for row in data["results"]:
        if row["status"] != "complete":
            continue
        print(f'| {row["benchmark"]} | {row["method"]} | {row["backbone"]} | {row["score"]:.2f} | {row["questions"]} |')
if __name__ == "__main__":
    main()
