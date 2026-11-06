"""Capture real Qwen / Mistral answers under matched memory conditions."""
import argparse
import copy
from datetime import date
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
    for row in rows:
        if isinstance(row, dict):
            if not all(isinstance(row.get(k), str) and row[k].strip() for k in ("body", "speaker", "date", "channel")):
                raise ValueError("Dated messages need body, speaker, date and channel")
            date.fromisoformat(row["date"])
            body = row["body"]
        else:
            body = row
        if not isinstance(body, str) or not body.strip() or len(body) > 1000:
            raise ValueError("Each observation must contain 1–1000 characters")
    return rows


def validate(case):
    for key in ("title", "speaker"):
        if not isinstance(case.get(key), str) or not case[key].strip():
            raise ValueError(f"Expected text for {key}")
    records = phase_notes(case, "source") + phase_notes(case, "update")
    dates = [date.fromisoformat(note["date"]) for note in records if isinstance(note, dict)]
    if dates != sorted(dates):
        raise ValueError("Dated messages must be chronological across the handoff")
    qs = case.get("questions")
    if not isinstance(qs, list) or not 1 <= len(qs) <= 8 or not all(isinstance(q, str) and q.strip() and len(q) <= 360 for q in qs):
        raise ValueError("Supply 1–8 questions, each up to 360 characters")
    labels = case.get("labels")
    if labels is not None and (not isinstance(labels, list) or len(labels) != len(qs) or not all(isinstance(x, str) and x.strip() for x in labels)):
        raise ValueError("Provide one label per question")


def capture(case, progress=None):
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

    assistant_mode = case.get("task_mode") == "project_assistant"

    def make_host(host):
        bridge = make_bridge(core, "locomo", host)
        if assistant_mode:
            # Use the same memory packet and host interface with a user-assistant
            # prompt instead of the short-answer benchmark rendering.
            from migro.backend.rd_benchmark import BenchmarkBridge
            bridge.render = BenchmarkBridge.render.__get__(bridge)
        return bridge

    def observation(i):
        note = texts[i]
        body, speaker = note, case["speaker"]
        stamp = "date not specified" if assistant_mode else f"{i+1} January, 2026"
        session = "source" if i < cut else "target"
        if isinstance(note, dict):
            body, speaker = note["body"], note["speaker"]
            stamp = date.fromisoformat(note["date"]).strftime("%d %B, %Y")
            session = note["date"]
        return dict(body=body, speaker=speaker, role="observed_human", date=stamp,
                    session=session, dialogue_id=f"note-{i+1}")

    def append(runtime, indices):
        rows = [observation(i) for i in indices]
        runtime.observe(rows)

    def measure(runtime, name, indices, receiver_writes):
        state = runtime.store.state
        before = state.serialize()
        system = "Answer briefly using the conversation memory."
        if assistant_mode:
            system = ("Help the user act on their project correspondence. Answer the specific request "
                      "in a few concise sentences. Respect recorded constraints and later revisions. "
                      "Distinguish proposed arrangements from confirmed bookings. If a detail is "
                      "missing, identify what to confirm. Do not claim to have taken external action.")
        queries = [dict(id=f"demo-{i}", benchmark="locomo", retrieval_query=q, system=system,
                        messages=[dict(role="user", content=q)]) for i, q in enumerate(case["questions"])]
        if progress:
            progress(name, 0, len(queries))
        if assistant_mode:
            # The project desk uses the same frozen reader/profile for all arms.
            # Independent questions can share a forward pass after the state is sealed.
            from migro.backend.decomem import make_reader
            cfg = runtime.config
            reader = make_reader(state, ranker, runtime.task, runtime.host)
            packets, slots = [], []
            limit = 256
            for query in queries:
                z, _, meta = reader.select_evidence(query["retrieval_query"], cfg["topk"],
                                                    cfg["source_token_cap"], cfg["mode"])
                selected = list(meta["slots"])
                while selected and len(runtime.bridge.build_item(query, z=z)[0]) + limit > runtime.bridge.context_limit:
                    selected.pop()
                    z, _ = reader.packet(selected)
                packets.append(z)
                slots.append([indices[int(j)] for j in selected])
            raw, cost = runtime.bridge.generate([{"query": q} for q in queries], latents=packets, max_new=limit)
            lengths = cost["generated_tokens_per_answer"]
            answers = [dict(text=text, slots=ids, generated_tokens=length, hit_output_limit=length >= limit)
                       for text, ids, length in zip(raw, slots, lengths)]
            runtime.bridge.connector.clear_query_cache()
        else:
            answers = []
            for i, query in enumerate(queries):
                if progress:
                    progress(name, i, len(queries))
                result = runtime.answer(query)
                answers.append(dict(text=result["answer"], slots=[indices[int(j)] for j in result["slots"]]))
        if state.serialize() != before:
            raise RuntimeError("Reading modified the memory")
        stages.append(dict(name=name, host=runtime.host, bytes=len(before), observations=len(indices),
                           note_ids=indices, receiver_writes=receiver_writes, answers=answers))
        if progress:
            progress(name, len(answers), len(answers))
        print(f"Recorded {name}", flush=True)

    def bindings(store):
        return [store.state.mem.read_slot(i, False).tolist() for i in range(store.observations)]

    with TemporaryDirectory() as directory:
        path = Path(directory) / "memory.bin"
        source_store = MemoryStore.create(core)
        source = MemoryRuntime(source_store, "A", "locomo", ranker, make_host("A"))
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
        bridge = make_host("C")
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
                answer_token_limit=256 if assistant_mode else None,
                source_model="Qwen3-4B-Instruct-2507", target_model="Mistral-7B-Instruct-v0.3",
                source_bindings_preserved=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=Path, default=HERE / "renovation.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--progress", type=Path, help="Optional live generation progress")
    args = parser.parse_args()

    def report(stage, done, total):
        path = args.progress
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(dict(stage=stage, done=done, total=total)))
        tmp.replace(path)
    trace = capture(json.loads(args.case.read_text()), report if args.progress else None)
    args.output.write_text(json.dumps(trace, ensure_ascii=False, indent=2) + "\n")
