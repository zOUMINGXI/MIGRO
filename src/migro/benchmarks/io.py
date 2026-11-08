"""Small file helpers for resumable benchmark entry points."""
import json
from pathlib import Path
from .._io import atomic_write_bytes

def read(path: str | Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def write(path: str | Path, value) -> None:
    atomic_write_bytes(path, (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))

def ensure_run_config(output: Path, config: dict, *, name="RUN_CONFIG.json",
                      artifacts=("states", "predictions.jsonl")) -> None:
    output = Path(output)
    path = output / name
    if path.exists():
        if read(path) != config:
            raise ValueError("Different run configuration in this output directory; use a separate output directory.")
        return
    if any((output / item).exists() for item in artifacts):
        raise ValueError("Existing run has no saved configuration; use a separate output directory.")
    write(path, config)

def predictions(path: Path, *, recover_incomplete=False) -> dict:
    path = Path(path)
    if not path.exists():
        return {}
    raw = path.read_bytes()
    lines = raw.splitlines(keepends=True)
    rows = []
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except (json.JSONDecodeError, UnicodeDecodeError):
            if not (recover_incomplete and i == len(lines)-1 and not line.endswith(b"\n")):
                raise
            # A terminated writer may leave an unfinished final append. Earlier
            # complete records remain intact; the missing answer is regenerated.
            raw = b"".join(lines[:i])
            atomic_write_bytes(path, raw)
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate prediction IDs")
    if recover_incomplete and raw and not raw.endswith(b"\n"):
        atomic_write_bytes(path, raw + b"\n")
    return {row["id"]: row for row in rows}

def append(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
