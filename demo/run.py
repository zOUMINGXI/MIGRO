"""Capture real Qwen / Mistral answers under matched memory conditions."""
import argparse
import copy
import gc
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

HERE = Path(__file__).resolve().parent


def phase_notes(case, phase):
    value = case.get(phase)
    rows = [value] if isinstance(value, str) else value
    if not isinstance(rows, list) or not 1 <= len(rows) <= 12:
        raise ValueError(f"Supply 1–12 observations for {phase}")
    if not all(isinstance(row, str) and row.strip() and len(row) <= 1000 for row in rows):
        raise ValueError("Each observation must contain 1–1000 characters")
    return rows


def validate(case):
    for key in ("title", "speaker"):
        if not isinstance(case.get(key), str) or not case[key].strip():
            raise ValueError(f"Expected text for {key}")
    phase_notes(case, "source")
    phase_notes(case, "update")
    qs = case.get("questions")
    if not isinstance(qs, list) or not 1 <= len(qs) <= 8 or not all(isinstance(q, str) and q.strip() and len(q) <= 360 for q in qs):
        raise ValueError("Supply 1–8 questions, each up to 360 characters")
    labels = case.get("labels")
    if labels is not None and (not isinstance(labels, list) or len(labels) != len(qs) or not all(isinstance(x, str) and x.strip() for x in labels)):
        raise ValueError("Provide one label per question")


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
    prefix = phase_notes(case, "source")
    suffix = phase_notes(case, "update")
    texts = prefix + suffix
    cut = len(prefix)
    stages = []
    old = list(range(cut))
    new = list(range(cut, len(texts)))

    def append(runtime, indices):
        rows = [dict(body=texts[i], speaker=case["speaker"], role="observed_human",
                     date=f"{i + 1} January, 2026", session="source" if i < cut else "target",
                     dialogue_id=f"note-{i + 1}") for i in indices]
        runtime.observe(rows)

    def measure(runtime, name, indices, receiver_writes):
        state = runtime.store.state
        before = state.serialize()
        answers = []
        for i, question in enumerate(case["questions"]):
            query = dict(id=f"demo-{i}", benchmark="locomo", retrieval_query=question,
                         system="Answer briefly using the conversation memory.",
                         messages=[dict(role="user", content=question)])
            result = runtime.answer(query)
            answers.append(dict(text=result["answer"], slots=[indices[int(j)] for j in result["slots"]]))
        if state.serialize() != before:
            raise RuntimeError("Reading modified the memory")
        stages.append(dict(name=name, host=runtime.host, bytes=len(before), observations=len(indices),
                           note_ids=indices, receiver_writes=receiver_writes, answers=answers))
        print(f"Recorded {name}", flush=True)

    def bindings(store):
        return [store.state.mem.read_slot(i, False).tolist() for i in range(store.observations)]

    with TemporaryDirectory() as directory:
        path = Path(directory) / "memory.bin"
        source_store = MemoryStore.create(core)
        source = MemoryRuntime(source_store, "A", "locomo", ranker, make_bridge(core, "locomo", "A"))
        append(source, old)
        measure(source, "source", old, 0)
        path.write_bytes(source_store.state.serialize())
        original_ids = bindings(source_store)
        original_meta = copy.deepcopy(source_store.state.meta)
        del source, source_store
        gc.collect()
        if torch.device(device).type == "cuda":
            torch.cuda.empty_cache()
        receiver = MemoryStore.from_snapshot(core, path.read_bytes())
        if receiver.state.serialize() != path.read_bytes():
            raise RuntimeError("Handoff changed the saved bytes")
        bridge = make_bridge(core, "locomo", "C")
        target = MemoryRuntime(receiver, "C", "locomo", ranker, bridge)
        measure(target, "handoff", old, 0)
        append(target, new)
        if bindings(receiver)[:cut] != original_ids or receiver.state.meta[:cut] != original_meta:
            raise RuntimeError("Append changed earlier records")
        measure(target, "append", old + new, len(new))
        for name, keep_old in [("new_only", False), ("rebuild", True)]:
            other = MemoryRuntime(MemoryStore.create(core), "C", "locomo", ranker, bridge)
            if keep_old:
                append(other, old)
            append(other, new)
            indices = old + new if keep_old else new
            measure(other, name, indices, len(indices))
    return dict(schema=2, case=case, stages=stages, device=device,
                source_model="Qwen3-4B-Instruct-2507", target_model="Mistral-7B-Instruct-v0.3",
                source_bindings_preserved=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=Path, default=HERE / "session.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    trace = capture(json.loads(args.case.read_text()))
    args.output.write_text(json.dumps(trace, ensure_ascii=False, indent=2) + "\n")
