"""Exercise the scoring command with deterministic upstream/judge stand-ins."""
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch
from migro.benchmarks import mab_score


class MABScoringTests(unittest.TestCase):
    def fixture(self, root, judged=False):
        work = root / "work"
        upstream = root / "upstream"
        output = root / "output"
        upstream.mkdir(); output.mkdir()
        manifest, predictions = [], []
        for i in range(4):
            cid = f"case{i}"
            sub = "longmemeval_s*" if judged and i == 0 else f"task{i}"
            folder = work / "cases" / cid
            folder.mkdir(parents=True)
            label = dict(id=f"q{i}", case_id=cid, answer="right", raw_question="Question",
                         question_type="single-session-user", sub_dataset=sub)
            (folder / "labels.json").write_text(json.dumps([label]))
            manifest.append(dict(case_id=cid, questions=1, sub_dataset=sub, category=f"ability{i}",
                                 metric="llm_judge_accuracy" if judged and i == 0 else "exact_match",
                                 config={}))
            predictions.append(dict(id=f"q{i}", case_id=cid, prediction="right", sub_dataset=sub))
        (work / "MANIFEST.json").write_text(json.dumps(manifest))
        (work / "assets").mkdir()
        (work / "assets/entity2id.json").write_text("{}")
        (output / "predictions.jsonl").write_text("".join(json.dumps(r)+"\n" for r in predictions))
        metric = types.ModuleType("utils.eval_other_utils")
        metric.post_process = lambda prediction, answer, config: ({"exact_match": float(prediction["output"] == answer)}, None)
        utils = types.ModuleType("utils"); utils.eval_other_utils = metric
        qa = types.ModuleType("llm_based_eval.longmem_qa_evaluate")
        qa.get_anscheck_prompt = lambda *args, **kwargs: "Judge this answer"
        summary = types.ModuleType("llm_based_eval.summarization_evaluate")
        llm = types.ModuleType("llm_based_eval"); llm.summarization_evaluate = summary
        recsys = types.ModuleType("migro.benchmarks.recsys_exact")
        recsys.find_nearest_movie = lambda *args, **kwargs: None
        modules = {"migro.benchmarks.recsys_exact": recsys, "utils": utils, "utils.eval_other_utils": metric, "llm_based_eval": llm,
                   "llm_based_eval.longmem_qa_evaluate": qa, "llm_based_eval.summarization_evaluate": summary}
        return work, upstream, output, predictions, modules

    def run_command(self, root, work, upstream, output, modules, relative=False):
        argv = ["migro-mab-score", "--work-dir", str(work.relative_to(root) if relative else work),
                "--upstream-root", str(upstream.relative_to(root) if relative else upstream),
                "--output", str(output.relative_to(root) if relative else output), "--judge-model", "new-judge"]
        cwd = Path.cwd()
        try:
            os.chdir(root)
            with patch.dict(sys.modules, modules), patch.object(sys, "argv", argv):
                mab_score.main()
        finally:
            os.chdir(cwd)

    def test_documented_relative_output_path(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            work, upstream, output, _, modules = self.fixture(root)
            self.run_command(root, work, upstream, output, modules, relative=True)
            self.assertTrue(json.loads((output/"RESULTS.json").read_text())["completed"])

    def test_duplicate_prediction_is_rejected_before_scoring(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            work, upstream, output, records, modules = self.fixture(root)
            with (output/"predictions.jsonl").open("a") as f:
                f.write(json.dumps(records[0])+"\n")
            with self.assertRaisesRegex(ValueError, "[Dd]uplicate"):
                self.run_command(root, work, upstream, output, modules)

    def test_changed_judge_does_not_reuse_old_verdict(self):
        from migro.benchmarks import judge_client
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            work, upstream, output, _, modules = self.fixture(root, judged=True)
            (output/"judgements.jsonl").write_text(json.dumps(dict(
                id="q0", prediction="right", score=0., judge_model="old-judge"))+"\n")
            response = {"choices": [{"message": {"content": "Yes"}}]}
            with patch.object(judge_client, "connection", return_value={"base_url":"https://example.invalid","api_key":"test"}), \
                 patch.object(judge_client, "post_json", return_value=response) as post:
                self.run_command(root, work, upstream, output, modules)
            self.assertEqual(post.call_count, 1)
            scored = json.loads((output/"scored.json").read_text())
            self.assertEqual(next(r["score"] for r in scored if r["id"] == "q0"), 1.)

    def test_mixed_read_budgets_are_not_one_scored_run(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            work, upstream, output, records, modules = self.fixture(root)
            for i, row in enumerate(records):
                row["source_token_cap"] = 2000 if i == 0 else 8000
            (output/"predictions.jsonl").write_text("".join(json.dumps(r)+"\n" for r in records))
            with self.assertRaisesRegex(ValueError, "Mixed generation"):
                self.run_command(root, work, upstream, output, modules)

    def test_matching_judge_reuses_completed_verdict(self):
        from migro.benchmarks import judge_client
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            work, upstream, output, _, modules = self.fixture(root, judged=True)
            (output/"judgements.jsonl").write_text(json.dumps(dict(
                id="q0", prediction="right", score=1., judge_model="new-judge"))+"\n")
            with patch.object(judge_client, "post_json") as post:
                self.run_command(root, work, upstream, output, modules)
            post.assert_not_called()
