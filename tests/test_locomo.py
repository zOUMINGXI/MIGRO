import math
import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from migro_eval import composite_from_components, summarize_scored
def cohort():
    return [
        dict(id=str(i), method="fixture", backbone="fixture", category=1 if i < 1540 else 5,
             score=1.0 if i < 1540 else 0.0)
        for i in range(1986)
    ]
class LoCoMoReportingTests(unittest.TestCase):
    def test_question_weighting_is_not_category_macro_average(self):
        result = summarize_scored(cohort())[0]
        self.assertAlmostEqual(result["score"], 100 * 1540 / 1986)
        self.assertNotEqual(result["score"], 50)
        self.assertTrue(result["complete"])
    def test_components_match_published_frozen_point(self):
        value = composite_from_components(50.45933709540633, 66.81614349775785)
        self.assertAlmostEqual(value, 54.13261788868366)
    def test_duplicate_question_rejected(self):
        rows = cohort()
        rows[-1]["id"] = rows[0]["id"]
        with self.assertRaises(ValueError):
            summarize_scored(rows)
    def test_partial_is_explicit(self):
        with self.assertRaises(ValueError):
            summarize_scored(cohort()[:2])
        result = summarize_scored(cohort()[:2], require_full=False)[0]
        self.assertFalse(result["complete"])
        self.assertEqual(result["questions"], 2)
    def test_units_and_nonfinite_values_rejected(self):
        for bad in (50.0, -0.1, math.inf, math.nan, True):
            rows = cohort()
            rows[0]["score"] = bad
            with self.assertRaises(ValueError):
                summarize_scored(rows)
if __name__ == "__main__":
    unittest.main()
