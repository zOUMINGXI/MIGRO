"""MIGRO source-prefix transfer, receiver continuation and full-history rebuilding."""
import argparse, gc, shutil, time
from pathlib import Path
from .io import read, write, predictions, append, ensure_run_config, atomic_write_bytes

ARMS = ("transfer_only", "transfer_continue", "target_rebuild")

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--phase", required=True, choices=["source", "target"])
    parser.add_argument("--target-host", choices=["B", "C"], default="C")
    args = parser.parse_args()
    import torch
    from ..memory import MemoryStore, MemoryRuntime, NativeBindingWriter, load_core
    from ..backend.decomem import make_bridge
    from ..backend.rd_evidence_reader import EvidenceRanker
    torch.set_num_threads(4); torch.manual_seed(7); torch.set_grad_enabled(False)
    manifest = read(args.manifest); root = args.manifest.resolve().parent
    ensure_run_config(args.output, {"manifest": manifest, "manifest_path": str(args.manifest.resolve()),
                                   "source_host": "A", "writer_window": 4, "writer_batch": 8},
                      artifacts=("source", *ARMS))
    if args.phase == "target":
        ensure_run_config(args.output, {"target_host": args.target_host}, name="TARGET_CONFIG.json", artifacts=ARMS)
    core = load_core("cuda"); host = "A" if args.phase == "source" else args.target_host
    bridge = make_bridge(core, "locomo", host)
    bridge.host.eval().requires_grad_(False); bridge.last_logits_only = True
    def rows(case):
        values = read(root / case["observations"])
        if not 0 <= case["cut"] <= len(values):
            raise ValueError("Invalid chronological cut")
        return values
    def build(path, values, prefix=None):
        path.mkdir(parents=True, exist_ok=True); snapshot = path / "state.bin"
        if (path / "WRITE.json").exists() and read(path / "WRITE.json").get("completed"):
            return
        store = MemoryStore.from_snapshot(core, snapshot.read_bytes()) if snapshot.exists() else (
            MemoryStore.from_snapshot(core, prefix.read_bytes()) if prefix else MemoryStore.create(core))
        offset = read(path / "START.json")["prefix_count"] if (path / "START.json").exists() else store.observations
        if not (path / "START.json").exists():
            write(path / "START.json", {"prefix_count": offset})
        done = store.observations - offset
        if not 0 <= done <= len(values):
            raise ValueError("Invalid resumed observation boundary")
        before = [store.state.mem.read_slot(i, False).tolist() for i in range(offset)]
        metadata = store.state.meta[:offset]
        writer = NativeBindingWriter(store.state, bridge, host, window=4)
        for i in range(done, len(values), 8):
            writer.observe(values[i:i+8])
            atomic_write_bytes(snapshot, store.state.serialize())
            write(args.output / "progress.json", {"stage": "write", "host": host,
                  "case": str(path.name), "done": min(i+8, len(values)), "total": len(values), "at": time.time()})
        if store.observations != offset + len(values) or store.state.meta[:offset] != metadata:
            raise RuntimeError("Continuation changed the prefix metadata")
        if before != [store.state.mem.read_slot(i, False).tolist() for i in range(offset)]:
            raise RuntimeError("Continuation changed prior source-token bindings")
        atomic_write_bytes(snapshot, store.state.serialize())
        write(path / "WRITE.json", {"completed": True, "host": host, "prefix_count": offset,
              "added_observations": len(values), "source_prefix_preserved": bool(prefix),
              "questions_seen_by_writer": False})
        del store, writer; gc.collect()
    if args.phase == "source":
        for case in manifest:
            build(args.output / "source" / case["case_id"], rows(case)[:case["cut"]])
        write(args.output / "SOURCE_COMPLETE.json", {"completed": True, "cases": len(manifest)})
        return
    if not (args.output / "SOURCE_COMPLETE.json").exists():
        raise ValueError("Complete the source phase before target-side construction")
    # Every target state is completed before opening any final question file.
    for case in manifest:
        cid = case["case_id"]; values = rows(case); prefix = args.output / "source" / cid / "state.bin"
        for arm in ARMS:
            destination = args.output / arm / "states" / cid
            if arm == "transfer_only":
                destination.mkdir(parents=True, exist_ok=True)
                atomic_write_bytes(destination / "state.bin", prefix.read_bytes())
            else:
                build(destination, values[case["cut"]:] if arm == "transfer_continue" else values,
                      prefix if arm == "transfer_continue" else None)
    write(args.output / "TARGET_BUILDS_COMPLETE.json", {"completed": True, "conditions": list(ARMS)})
    ranker = EvidenceRanker()
    for arm in ARMS:
        output = args.output / arm / "predictions.jsonl"; saved = predictions(output, recover_incomplete=True); expected = set()
        for case in manifest:
            cid = case["case_id"]; snapshot = args.output / arm / "states" / cid / "state.bin"
            store = MemoryStore.from_snapshot(core, snapshot.read_bytes())
            runtime = MemoryRuntime(store, host, "locomo", ranker, bridge)
            immutable = store.state.serialize()
            queries = read(root / case["questions"]); expected.update(q["id"] for q in queries)
            for query in queries:
                if query["id"] in saved:
                    continue
                result = runtime.answer(query); result["prediction"] = result.pop("answer")
                row = dict(result, id=query["id"], case_id=cid, arm=arm, method="migro",
                           source_host="A", target_host=host, final_full_history_labels=True)
                append(output, row); saved[row["id"]] = row
            if immutable != store.state.serialize():
                raise RuntimeError("Final questions mutated memory")
            del runtime, store; bridge.connector.clear_query_cache(); gc.collect()
        if set(saved) != expected:
            raise ValueError("Incomplete final-history question cohort")
    write(args.output / "GENERATION_COMPLETE.json", {"completed": True, "conditions": list(ARMS),
          "target_host": host, "score_against": "complete final-history labels"})

if __name__ == "__main__":
    main()
