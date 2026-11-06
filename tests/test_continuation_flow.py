"""Exercise chronology, sealed questions and continuation restart without GPU generation."""
import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch required")
class ContinuationFlowTests(unittest.TestCase):
    def test_three_arms_preserve_prefix_and_restart_without_duplicate_answers(self):
        import torch
        from migro.benchmarks import continue_write as cli
        class Store:
            def __init__(self, rows=None):
                self.rows = list(rows or [])
                self.state = types.SimpleNamespace(
                    meta=self.rows, mem=types.SimpleNamespace(read_slot=self.read_slot),
                    serialize=lambda: json.dumps(self.rows).encode())
            @property
            def observations(self):
                return len(self.rows)
            def read_slot(self, index, _):
                return torch.tensor([ord(c) for c in self.rows[index]["body"]])
            @classmethod
            def create(cls, core):
                return cls()
            @classmethod
            def from_snapshot(cls, core, blob):
                return cls(json.loads(blob))
        class Writer:
            def __init__(self, state, bridge, host, window):
                self.state = state
            def observe(self, rows):
                self.state.meta.extend(rows)
        class Runtime:
            def __init__(self, store, *args):
                self.store = store
            def answer(self, query):
                return {"answer": str(self.store.observations)}
        class Host:
            def eval(self):
                return self
            def requires_grad_(self, value):
                return self
        bridge = types.SimpleNamespace(host=Host(),
                     connector=types.SimpleNamespace(clear_query_cache=lambda: None))
        memory = types.ModuleType("migro.memory")
        memory.MemoryStore=Store; memory.MemoryRuntime=Runtime; memory.NativeBindingWriter=Writer
        memory.load_core=lambda device: object()
        backend = types.ModuleType("migro.backend.decomem")
        backend.make_bridge=lambda *args: bridge
        reader = types.ModuleType("migro.backend.rd_evidence_reader")
        reader.EvidenceRanker=lambda: object()
        modules={"migro.memory":memory, "migro.backend.decomem":backend,
                 "migro.backend.rd_evidence_reader":reader}
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); output=root/"out"
            observations=[{"body":str(i),"dialogue_id":str(i)} for i in range(5)]
            (root/"observations.json").write_text(json.dumps(observations))
            (root/"questions.json").write_text(json.dumps([{"id":"q1"}]))
            manifest=root/"manifest.json"
            manifest.write_text(json.dumps([{"case_id":"case","observations":"observations.json",
                                             "questions":"questions.json","cut":2}]))
            original_read=cli.read
            def sealed_read(path):
                if Path(path).name=="questions.json":
                    self.assertTrue((output/"TARGET_BUILDS_COMPLETE.json").exists())
                    self.assertTrue(all((output/arm/"states/case/state.bin").exists() for arm in cli.ARMS))
                return original_read(path)
            with patch.dict(sys.modules, modules), patch.object(cli,"read",sealed_read):
                for phase in ["source","target","target"]:
                    with patch.object(sys,"argv",["migro-continue","--manifest",str(manifest),
                                                  "--output",str(output),"--phase",phase]):
                        cli.main()
            expected={"transfer_only":2,"transfer_continue":5,"target_rebuild":5}
            for arm, size in expected.items():
                state=json.loads((output/arm/"states/case/state.bin").read_bytes())
                self.assertEqual(len(state),size)
                self.assertEqual(state[:2],observations[:2])
                self.assertEqual(len((output/arm/"predictions.jsonl").read_text().splitlines()),1)
            self.assertEqual(json.loads((output/"source/case/state.bin").read_bytes()),observations[:2])

if __name__=="__main__":
    unittest.main()
