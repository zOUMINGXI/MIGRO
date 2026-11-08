import unittest
from migro.benchmarks.format_corrected_metrics import extract
from migro.benchmarks.query_intent import retrieval_intent

class MABProtocolTests(unittest.TestCase):
    def test_complete_label_and_json_answer_only(self):
        self.assertEqual(extract("Answer: label: 42", "icl_nlu"), "42")
        self.assertIsNone(extract("42 or 43", "icl_nlu"))
        self.assertIsNone(extract("The answer is 42 because...", "icl_nlu"))
        self.assertEqual(extract('{"answer":"Ada"}', "detective_qa"), "Ada")
        self.assertIsNone(extract('{"answer":12}', "detective_qa"))
        self.assertIsNone(extract('{"answer":"Ada"', "detective_qa"))

    def test_retrieval_intent_removes_demonstrations_without_gold(self):
        query = "Solve the case. For example: Q: old A: old. Now Answer the Question: Who left?"
        self.assertEqual(retrieval_intent(query, "detective_qa"), "Who left?")
        self.assertEqual(retrieval_intent(query, "eventqa_full"), query)
        summary = "Summarize carefully. For example: old summary. Now summarize the book below."
        self.assertEqual(retrieval_intent(summary, "infbench_sum_eng_shots2"),
                         "Summarize carefully.\n\nNow summarize the book below.")


class QuestionMetadataTests(unittest.TestCase):
    def test_original_question_ids_survive_nested_and_top_level_metadata(self):
        from migro.benchmarks.mab_prepare import question_field
        for row in [{"question_ids":["q1","q2_abs"]},
                    {"metadata":{"question_ids":["q1","q2_abs"]}}]:
            self.assertEqual(question_field(row,"question_ids",1), "q2_abs")
        self.assertIsNone(question_field({}, "question_ids", 0))

if __name__ == "__main__":
    unittest.main()
