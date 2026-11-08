"""Resume contracts and interrupted checkpoint behavior."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from migro.benchmarks import io


class ResumeIOTests(unittest.TestCase):
    def test_budget_or_shard_change_cannot_relabel_existing_output(self):
        with tempfile.TemporaryDirectory() as d:
            output = Path(d)/"run"
            original = {"source_token_cap":2000, "shards":2, "shard_index":0}
            io.ensure_run_config(output, original)
            io.ensure_run_config(output, dict(original))
            for changed in [dict(original, source_token_cap=8000), dict(original, shard_index=1)]:
                with self.assertRaisesRegex(ValueError, "[Dd]ifferent"):
                    io.ensure_run_config(output, changed)
            self.assertEqual(io.read(output/"RUN_CONFIG.json"), original)

    def test_recover_only_an_unfinished_final_record(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/"predictions.jsonl"
            first = json.dumps({"id":"a","prediction":"one"})+"\n"
            path.write_text(first+'{"id":"b","prediction":')
            rows = io.predictions(path, recover_incomplete=True)
            self.assertEqual(list(rows), ["a"])
            io.append(path, {"id":"b","prediction":"two"})
            self.assertEqual(len(io.predictions(path)), 2)
            path.write_bytes(first.encode() + b'{"id":"b","prediction":"' + "京都".encode()[:1])
            self.assertEqual(list(io.predictions(path, recover_incomplete=True)), ["a"])
            path.write_text('{"broken":\n'+first)
            with self.assertRaises(ValueError):
                io.predictions(path, recover_incomplete=True)

    def test_interrupted_snapshot_preserves_previous_file(self):
        from migro._io import atomic_write_bytes
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/"state.bin"
            atomic_write_bytes(path, b"last-complete-state")
            def interrupted(target, data):
                with target.open("wb") as stream:
                    stream.write(data[:3])
                raise OSError("simulated interruption")
            with patch.object(Path, "write_bytes", interrupted):
                with self.assertRaises(OSError):
                    atomic_write_bytes(path, b"replacement")
            self.assertEqual(path.read_bytes(), b"last-complete-state")
            atomic_write_bytes(path, b"replacement")
            self.assertEqual(path.read_bytes(), b"replacement")
