import sys
import json
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from migro.controls import build_bm25_state, build_recent_state, select_bm25, packet_source_ids

def rows():
    return [dict(slot=10, text="Ada visited Kyoto", source_ids=[10, 11], meta={"speaker": "Ada"}),
            dict(slot=20, text="Ada works in London", source_ids=[20, 21]),
            dict(slot=30, text="The final observation", source_ids=[30, 31])]

class MemoryControlTests(unittest.TestCase):
    def test_portable_json_roundtrip_preserves_retrieval_and_exact_ids(self):
        source = rows()
        state = json.loads(json.dumps(build_bm25_state(source)))
        slots = select_bm25(state, "Kyoto", budget=3)
        self.assertEqual(slots, [10])
        self.assertEqual(packet_source_ids(state, slots, separator_ids=[99]), [10, 11, 99])
        self.assertNotIn("text", state["rows"][0])
        self.assertEqual(source[0]["text"], "Ada visited Kyoto")

    def test_no_match_prefers_recent_but_delivery_is_chronological(self):
        state = build_bm25_state(rows())
        self.assertEqual(select_bm25(state, "unseen", budget=6), [20, 30])

    def test_recent_is_a_contiguous_complete_suffix(self):
        state = build_recent_state(rows(), budget=6)
        self.assertEqual([r["slot"] for r in state["rows"]], [20, 30])
        self.assertEqual(packet_source_ids(state, [20, 30], separator_ids=[99]),
                         [20, 21, 99, 30, 31, 99])

    def test_oversized_latest_is_not_replaced_by_old_history(self):
        source = rows()
        source[-1]["source_ids"] = list(range(20))
        self.assertEqual(build_recent_state(source, budget=6)["rows"], [])

    def test_zero_budget_and_duplicate_slots(self):
        self.assertEqual(select_bm25(build_bm25_state(rows()), "Kyoto", budget=0), [])
        source = rows()
        source[-1]["slot"] = source[0]["slot"]
        with self.assertRaises(ValueError):
            build_bm25_state(source)
