import importlib.util
import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from migro_eval.ablations import paired_summary

def cohort():
    return [dict(id="a1", case_id="a", host="same", score=0.0),
            dict(id="a2", case_id="a", host="same", score=0.0),
            dict(id="a3", case_id="a", host="same", score=0.0),
            dict(id="b1", case_id="b", host="same", score=0.0)]

@unittest.skipUnless(importlib.util.find_spec("numpy"), "optional analysis dependency")
class PairedAnalysisTests(unittest.TestCase):
    def test_constant_effect_has_point_interval(self):
        full = cohort()
        other = [dict(r, score=0.25) for r in full]
        result = paired_summary({"full": full, "other": other}, reference="full",
                                expected_questions=4, bootstrap=100)
        self.assertEqual(result["paired"]["other"]["delta_vs_reference"], 25.0)
        self.assertEqual(result["paired"]["other"]["conversation_cluster_95ci"], [25.0, 25.0])

    def test_question_weighting_and_whole_cluster_resampling(self):
        full = cohort()
        other = [dict(r, score=1.0 if r["case_id"] == "a" else 0.0) for r in full]
        result = paired_summary({"full": full, "other": other}, reference="full",
                                expected_questions=4, bootstrap=1000)
        self.assertEqual(result["paired"]["other"]["delta_vs_reference"], 75.0)
        self.assertEqual(result["conversations"], 2)
        self.assertEqual(result["paired"]["other"]["conversation_cluster_95ci"], [0.0, 100.0])

    def test_incomplete_mismatched_or_cross_backbone_pairs_rejected(self):
        full = cohort()
        variants = [full[:-1], [dict(r, host="different") for r in full],
                    [dict(r, id="changed") if i == 0 else r for i, r in enumerate(full)]]
        for bad in variants:
            with self.assertRaises(ValueError):
                paired_summary({"full": full, "bad": bad}, reference="full",
                               expected_questions=4, bootstrap=10)
