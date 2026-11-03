"""Summarize already-scored ablation conditions for one backbone."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from migro_eval.ablations import paired_summary

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", action="append", required=True, metavar="NAME=PATH")
    parser.add_argument("--reference", required=True)
    parser.add_argument("--expected-questions", type=int, default=1986)
    parser.add_argument("--bootstrap", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    conditions = {}
    for spec in args.condition:
        name, path = spec.split("=", 1)
        if name in conditions:
            parser.error(f"Repeated condition: {name}")
        conditions[name] = json.loads(Path(path).read_text())
    result = paired_summary(conditions, reference=args.reference,
                            expected_questions=args.expected_questions,
                            bootstrap=args.bootstrap, seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")

if __name__ == "__main__":
    main()
