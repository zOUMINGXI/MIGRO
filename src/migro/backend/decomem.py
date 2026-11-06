"""DeCoMem: Decoupled and Portable Long-Term Memory for Language Models."""
import copy
from pathlib import Path
from .storage_v11 import StorageV11,select_archive
from .rd_value_packed import verify
from .decomem_archive import MAGIC,encode,restore

NAME="DeCoMem"
EXPANSION="Decoupled and Portable Long-Term Memory for Language Models"
VERSION="1.0"
import os
from .paths import asset_path
PROFILE_PATH=Path(os.environ.get("MIGRO_PROFILES",str(Path(__file__).with_name("profiles.json"))))

class DeCoMem(StorageV11):
    @classmethod
    def from_snapshot(cls,core,blob,*,coder=None):
        if blob.startswith(MAGIC):return cls(restore(core,blob,coder),blob)
        return super().from_snapshot(core,blob)

    def compact(self,*,coder=None,qwen_already_deployed=True):
        """Choose the smallest exact archive. Timing never enters this decision.

        Supplying coder adds the shared frozen Qwen conditional-content candidate.
        Decode verification is required before selection. If Qwen is not deployed,
        its added model bytes enter the decision. Legacy formats remain readable.
        """
        retained=self._snapshot if not (self._snapshot and self._snapshot.startswith(MAGIC) and not qwen_already_deployed) else None
        literal,info=select_archive(self.state,retained)
        choices={"lossless_archive":literal}
        costs={"lossless_archive":len(literal)}
        detail={}
        if coder is not None and self.observations:
            neural,detail=encode(self.state,coder)
            candidate=restore(self.state.core,neural,coder)
            verify(self.state,candidate)
            choices["causal_content"]=neural
            from .content_rate_selector import QWEN_DIRECTORY_BYTES
            costs["causal_content"]=len(neural)+(0 if qwen_already_deployed else QWEN_DIRECTORY_BYTES)
        # A retained neural archive still requires its decoder model.
        if literal.startswith(MAGIC) and not qwen_already_deployed:
            from .content_rate_selector import QWEN_DIRECTORY_BYTES
            costs["lossless_archive"]+=QWEN_DIRECTORY_BYTES
        selected=min(costs,key=lambda k:(costs[k],k))
        self._snapshot=choices[selected]
        self._selection=dict(selected=selected,actual_snapshot_bytes=len(self._snapshot),
            candidate_objective_bytes=costs,latency_used_for_selection=False,
            exact_neural_candidate_verified=bool(detail),neural=detail,literal=info)
        return self._selection

    def handoff(self,target_core=None,*,coder=None):
        core=self.state.core if target_core is None else target_core
        return type(self).from_snapshot(core,self.snapshot(),coder=coder)

def profile(task,host):
    import json
    if host not in ("A","B","C"):raise ValueError(host)
    cfg=json.loads(PROFILE_PATH.read_text())
    return copy.deepcopy(cfg["reader_profiles"][task][host])

def make_reader(state,ranker,task,host):
    from .rd_evidence_reader import EvidenceReader
    from .rd_bound_evidence_reader import BoundEvidenceReader
    from .rd_role_coverage import RoleCoverageReader
    factories=dict(support=EvidenceReader,bound=BoundEvidenceReader,role_coverage=RoleCoverageReader)
    return factories[profile(task,host)["reader"]](state,ranker)

def make_bridge(core,task,host):
    from .rd_third_host import ThirdHostBridge
    from .rd_benchmark import BenchmarkBridge
    cfg=profile(task,host)
    if cfg["bridge"]=="coverage":
        class CoverageBridge(ThirdHostBridge):
            def render(self,query,content=None,soft=False):
                row=copy.deepcopy(query)
                row["system"]=(row.get("system") or "")+"\n"+COVERAGE_POLICY
                return BenchmarkBridge.render(self,row,content,soft)
        return CoverageBridge(core,host)
    bridge=ThirdHostBridge(core,host,cfg.get("checkpoint"))
    if cfg["bridge"]=="phi_boundary":
        import torch
        from .rd_boundary_interface import BoundaryTransport,build_boundary_item
        from types import MethodType
        base=bridge.connector.base
        bridge.connector=BoundaryTransport(base,bridge.host.config.hidden_size).to(bridge.device).eval()
        cp=torch.load(asset_path(cfg["boundary_checkpoint"]),map_location=bridge.device,weights_only=True)
        bridge.connector.adapter.load_state_dict(cp["adapter"])
        original=bridge.build_item
        def build(self,row,z=None,text=None):
            return build_boundary_item(self,row,z) if z is not None and len(z) else original(row,z,text)
        bridge.build_item=MethodType(build,bridge)
    return bridge

COVERAGE_POLICY=(
    "Answer the question using the recovered conversation evidence. "
    "Give every requested item or part, including the relevant names, values, and dates. "
    "Keep statements attached to their speaker and time. Use the version requested by the question. "
    "Separate the user's own facts from an assistant's suggestions. "
    "For a factual detail absent from the evidence, explicitly say it cannot be determined. "
    "Use a concise sentence or a short list. Include only calculations needed for the requested result. "
    "Do not repeat the question, quote long passages, or add speculative background."
)

class DeCoMemRuntime:
    """Frozen host access to one shared state; host switching never rewrites history."""
    def __init__(self,store,host="A",task="general",ranker=None,bridge=None,reader=None):
        from .rd_evidence_reader import EvidenceRanker
        self.reader=reader  # Optional prebuilt reader for a sealed snapshot.
        self.store=store;self.host=host;self.task=task;self.config=profile(task,host)
        self.bridge=bridge or make_bridge(store.state.core,task,host)
        self.ranker=ranker or EvidenceRanker()

    def answer(self,query):
        from .rd_evidence_reader import options_from_prompt,selective_null
        reader=self.reader if self.reader is not None else make_reader(self.store.state,self.ranker,self.task,self.host)
        cfg=self.config
        z,text,md=reader.select_evidence(query["retrieval_query"],cfg["topk"],cfg["source_token_cap"],cfg["mode"])
        blind=copy.deepcopy(query)
        original=blind["messages"][-1]["content"]
        prefix=original.split("Question:")[0] if "Question:" in original else ""
        blind["messages"][-1]["content"]=prefix+"Question: "+query["retrieval_query"]+"\nIf evidence does not support the requested detail for the named person, answer No information available."
        option=options_from_prompt(query["messages"][-1]["content"]) is not None
        check=[query,blind] if option and cfg["answer_policy"]=="strict_option_blind" else [query]
        while md["slots"] and max(len(self.bridge.build_item(q,z=z)[0]) for q in check)+cfg["max_new_tokens"]>self.bridge.context_limit:
            md["slots"].pop();z,text=reader.packet(md["slots"])
        ys,cost=self.bridge.generate([dict(query=query)],latents=[z],max_new=cfg["max_new_tokens"])
        raw=ys[0];recall=raw
        if option and cfg["answer_policy"]=="strict_option_blind":
            recall=self.bridge.generate([dict(query=blind)],latents=[z],max_new=cfg["max_new_tokens"])[0][0]
        answer=selective_null(query["retrieval_query"],raw,recall,reader.speakers,True)[0] if cfg["answer_policy"].startswith("strict") else raw
        self.bridge.connector.clear_query_cache()
        return dict(answer=answer,raw_answer=raw,recall=recall,slots=md["slots"],evidence=text,cost=cost,
                    host=self.host,task=self.task,generation_calls=2 if option and cfg["answer_policy"]=="strict_option_blind" else 1,original_history_replayed=False)

    def observe(self,observations):
        from .rd_native_writer import NativeBindingWriter
        rows=list(observations)
        if not rows:return dict(native_forwards=0)
        self.reader=None
        self.store._snapshot=None;self.store._selection=None;self.store.release_reader()
        writer=NativeBindingWriter(self.store.state,self.bridge,self.host)
        writer.observe(rows)
        return dict(native_forwards=writer.forwards,native_input_tokens=writer.tokens,
                    prior_window=writer.window,full_history_replayed=False)
