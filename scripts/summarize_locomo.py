"""Summarize existing scored records; this script does not run inference."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from migro_eval import summarize_scored
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="JSON array or JSONL scored records")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    text = args.input.read_text()
    records = [json.loads(line) for line in text.splitlines() if line.strip()] if args.input.suffix == ".jsonl" else json.loads(text)
    if not isinstance(records, list):
        raise ValueError("Expected a JSON array or JSONL records")
    result = summarize_scored(records, require_full=not args.allow_partial)
    rendered = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    else:
        print(rendered, end="")
if __name__ == "__main__":
    main()
