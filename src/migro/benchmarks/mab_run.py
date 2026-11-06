"""Incremental MIGRO MemoryAgentBench generation from a prepared work directory."""
import argparse, copy, gc, time
from pathlib import Path
from .io import read, write, predictions, append

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--source-token-cap", type=int, choices=[2000, 8000], default=2000)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    args = parser.parse_args()
    if not 0 <= args.shard_index < args.shards:
        parser.error("Require 0 <= shard-index < shards")
    import torch
    from types import MethodType
    from ..memory import MemoryStore, MemoryRuntime, load_core
    from ..backend.rd_benchmark import BenchmarkBridge, CAPS
    from ..backend.rd_third_host import ThirdHostBridge
    from ..backend.rd_evidence_reader import EvidenceRanker
    from .query_intent import retrieval_intent
    torch.set_num_threads(4); torch.manual_seed(7); torch.set_grad_enabled(False)
    manifest = read(args.work_dir / "MANIFEST.json")
    cases = [row for i, row in enumerate(manifest) if i % args.shards == args.shard_index]
    args.output.mkdir(parents=True, exist_ok=True)
    write(args.output / "RUN_CONFIG.json", {
        "source_token_cap": args.source_token_cap, "shard_index": args.shard_index,
        "shards": args.shards, "writer_window": 4, "incremental_turn_batch": 1,
        "retrieval_query_limit": 128, "full_generation_prompt_unchanged": True,
        "raw_history_at_answer": False, "new_calibration": False})
    core = load_core("cuda")
    bridge = ThirdHostBridge(core, "A", None)
    bridge.render = MethodType(BenchmarkBridge.render, bridge)
    bridge.context_limit = min(131072, bridge.host.config.max_position_embeddings)
    bridge.last_logits_only = True
    bridge.host.eval().requires_grad_(False); bridge.connector.eval().requires_grad_(False)
    class BoundedQueryRanker(EvidenceRanker):
        def score(self, query, passages):
            ids = self.tokenizer.encode(query, add_special_tokens=False, truncation=True, max_length=128)
            return super().score(self.tokenizer.decode(ids, skip_special_tokens=True), passages)
    ranker = BoundedQueryRanker()
    output = args.output / "predictions.jsonl"; saved = predictions(output); expected = set()
    for meta in cases:
        cid = meta["case_id"]; folder = args.work_dir / "cases" / cid
        destination = args.output / "states" / cid; destination.mkdir(parents=True, exist_ok=True)
        snapshot = destination / "state.bin"
        store = MemoryStore.from_snapshot(core, snapshot.read_bytes()) if snapshot.exists() else MemoryStore.create(core)
        # Future questions and labels are not opened while the native writer runs.
        observations = read(folder / "memory_units.json")
        if not 0 <= store.observations <= len(observations):
            raise ValueError("Saved state does not match observation count")
        runtime = MemoryRuntime(store, "A", "locomo", ranker, bridge)
        for i in range(store.observations, len(observations)):
            runtime.observe(observations[i:i+1])
            write(args.output / "progress.json", {"stage": "write", "case_id": cid,
                  "done": i+1, "total": len(observations), "at": time.time()})
            if (i+1) % 32 == 0:
                snapshot.write_bytes(store.state.serialize())
        snapshot.write_bytes(store.state.serialize())
        write(destination / "WRITE.json", {"completed": True, "observations": store.observations,
              "questions_or_labels_seen_by_writer": False, "writer_window": 4})
        del observations
        CAPS["memoryagentbench"] = meta["generation_max_length"]
        runtime.config.update(max_new_tokens=meta["generation_max_length"],
                              source_token_cap=args.source_token_cap, answer_policy="original")
        immutable = store.state.serialize()
        queries = read(folder / "questions.json")
        expected.update(q["id"] for q in queries)
        for original in queries:
            if original["id"] in saved:
                continue
            query = copy.deepcopy(original)
            query["retrieval_query"] = retrieval_intent(original["retrieval_query"], meta["sub_dataset"])
            if bridge.render(query) != bridge.render(original):
                raise RuntimeError("Retrieval intent modified the generation prompt")
            result = runtime.answer(query); result["prediction"] = result.pop("answer")
            row = dict(result, id=query["id"], case_id=cid, benchmark="memoryagentbench",
                       method="migro", host="A", sub_dataset=meta["sub_dataset"],
                       category=meta["category"], max_new_tokens=meta["generation_max_length"],
                       source_token_cap=args.source_token_cap, raw_history_at_answer=False)
            append(output, row); saved[row["id"]] = row
            write(args.output / "progress.json", {"stage": "answer", "responses": len(saved),
                                                "case_id": cid, "at": time.time()})
        if immutable != store.state.serialize():
            raise RuntimeError("Answering mutated the saved memory")
        del runtime, store, immutable
        bridge.connector.clear_query_cache(); gc.collect()
    if set(saved) != expected:
        raise ValueError("Predictions do not match this shard's complete question cohort")
    write(args.output / "GENERATION_COMPLETE.json", {"completed": True,
          "questions": len(saved), "contexts": len(cases), "shards": args.shards})

if __name__ == "__main__":
    main()
