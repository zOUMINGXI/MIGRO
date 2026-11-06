"""Record MIGRO's native Qwen → Mistral handoff, including verbatim model answers."""
import argparse
import copy
import gc
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

HERE = Path(__file__).resolve().parent


def validate(case):
    for key in ("title", "speaker", "source", "update"):
        if not isinstance(case.get(key), str) or not case[key].strip():
            raise ValueError(f"Expected nonempty text for {key}")
    questions = case.get("questions")
    if not isinstance(questions, list) or len(questions) != 3 or not all(isinstance(q, str) and q.strip() for q in questions):
        raise ValueError("Supply three questions")


def capture(case):
    import torch
    from migro.memory import MemoryStore, MemoryRuntime, load_core
    from migro.backend.decomem import make_bridge
    from migro.backend.rd_evidence_reader import EvidenceRanker

    validate(case)
    device = os.environ.get("MIGRO_DEVICE", "cuda")
    torch.set_num_threads(8)
    torch.manual_seed(7)
    torch.set_grad_enabled(False)
    core = load_core(device)
    ranker = EvidenceRanker(device=device)
    store = MemoryStore.create(core)
    stages = []

    def append(runtime, text, index):
        runtime.observe([dict(body=text, speaker=case["speaker"], role="observed_human",
                              date=f"{index + 1} January, 2026", session=f"session_{index + 1}",
                              dialogue_id=f"s{index + 1}:0")])

    def measure(runtime, name):
        before = runtime.store.state.serialize()
        answers = []
        for i, question in enumerate(case["questions"]):
            query = dict(id=f"demo-{i}", benchmark="locomo", retrieval_query=question,
                         system="Answer briefly using the conversation memory.",
                         messages=[dict(role="user", content=question)])
            result = runtime.answer(query)
            answers.append(dict(text=result["answer"], slots=list(map(int, result["slots"]))))
        if runtime.store.state.serialize() != before:
            raise RuntimeError("Read changed the stored state")
        stages.append(dict(name=name, host=runtime.host, observations=runtime.store.observations,
                           bytes=len(before), answers=answers))
        print(f"Recorded {name}", flush=True)

    def bindings(store):
        return [store.state.mem.read_slot(i, False).tolist() for i in range(store.observations)]

    with TemporaryDirectory() as directory:
        path = Path(directory) / "memory.bin"
        source = MemoryRuntime(store, "A", "locomo", ranker, make_bridge(core, "locomo", "A"))
        append(source, case["source"], 0)
        measure(source, "source")
        path.write_bytes(store.state.serialize())
        original_ids, original_meta = bindings(store), copy.deepcopy(store.state.meta)
        del source, store
        gc.collect()
        if torch.device(device).type == "cuda":
            torch.cuda.empty_cache()
        receiver = MemoryStore.from_snapshot(core, path.read_bytes())
        if receiver.state.serialize() != path.read_bytes():
            raise RuntimeError("Snapshot changed across handoff")
        target = MemoryRuntime(receiver, "C", "locomo", ranker, make_bridge(core, "locomo", "C"))
        measure(target, "handoff")
        append(target, case["update"], 1)
        if bindings(receiver)[:len(original_ids)] != original_ids or receiver.state.meta[:len(original_meta)] != original_meta:
            raise RuntimeError("Receiver changed the source observation")
        measure(target, "append")
    return dict(schema=1, case=case, stages=stages, device=device,
                source_model="Qwen3-4B-Instruct-2507", target_model="Mistral-7B-Instruct-v0.3",
                source_bindings_preserved=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=Path, default=HERE / "session.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    trace = capture(json.loads(args.case.read_text()))
    args.output.write_text(json.dumps(trace, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
