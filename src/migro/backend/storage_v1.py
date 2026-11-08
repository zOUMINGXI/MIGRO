"""EviWrite Storage v1: frozen shared addresses and exact observation bindings.

The version freezes an API and lossless snapshot policy, not a new memory model.
Readers and host interfaces retain their independent benchmark configurations.
All original observations and versions survive. No precision allocation or
boundary compaction is enabled here.
"""
from __future__ import annotations
from .paths import asset_path
from .._io import atomic_write_bytes

from pathlib import Path
from typing import Callable, Iterable

import torch

from .rd_dialogue import DialogueState
from .rd_evidence_reader import EvidenceReader
from .rd_runtime_accounting import owned_bytes, reader_ledger, share_public_tables
from .rd_shared_anchors import SharedAnchorMemory
from .rd_robust_residual import STORAGE_MAGIC, unpack_storage
from .rd_value_packed import restore as restore_snapshot, select

PROFILE = {
    "name": "eviwrite-storage-v1",
    "schema": 1,
    "address": "existing shared 384D roots / 64D innovations",
    "root_precision": "native 4 bit with FP16 scales",
    "innovation_precision": "native INT8 with FP16 scales",
    "bindings": "exact ordered source-token IDs",
    "retention": "all observations and intermediate versions",
    "snapshot_policy": "minimum actual bytes among existing and four lossless value codecs, each with the same optional outer compression",
    "codec_choices": ["original", "z_tokens", "z_byte_planes", "x_tokens", "x_byte_planes"],
    "experimental_precision_allocation": False,
    "experimental_boundary_compaction": False,
    "backbone_updates": 0,
    "new_weights": 0,
    "runtime": "decoded active state plus optional reader cache; not compressed RAM",
}


class StorageV1:
    """One independent, host-neutral state with an explicit observe/read boundary.

    Use observe() for mutations, so reader and snapshot caches are invalidated.
    The state property supports existing readers and is read-only to callers.
    snapshot() bytes can be opened by the existing restore_any() reader.
    """

    def __init__(self, state: DialogueState, snapshot: bytes | None = None):
        if not isinstance(state.mem, SharedAnchorMemory):
            raise TypeError("Storage v1 requires native fixed shared-anchor geometry")
        self._state = state
        share_public_tables(state.mem)
        self._snapshot = snapshot
        self._reader = None
        self._reader_key = None
        self._selection = None

    @classmethod
    def create(cls, core: torch.nn.Module, key_mode: str = "content") -> StorageV1:
        return cls(DialogueState(core, key_mode))

    @classmethod
    def from_snapshot(cls, core: torch.nn.Module, blob: bytes) -> StorageV1:
        payload = blob
        while payload.startswith(STORAGE_MAGIC):
            payload = unpack_storage(payload)
        return cls(restore_snapshot(core, payload), blob)

    @classmethod
    def load(cls, core: torch.nn.Module, path: str | Path) -> StorageV1:
        return cls.from_snapshot(core, Path(path).read_bytes())

    @property
    def state(self) -> DialogueState:
        return self._state

    @property
    def observations(self) -> int:
        return self._state.mem.n

    def observe(self, observations: Iterable[dict], writer: str = "canonical",
                links: list[int | None] | None = None) -> None:
        rows = list(observations)
        if not rows:
            return
        if links is not None and len(links) != len(rows):
            raise ValueError("One reference entry is required per observation")
        self._snapshot = None
        self._selection = None
        self.release_reader()
        self._state.append_many(rows, writer=writer, links=links)

    def snapshot(self) -> bytes:
        """Return stable bytes; a restored snapshot remains unchanged until a write."""
        if self._snapshot is None:
            self._snapshot, self._selection = select(self._state)
        return self._snapshot

    def compact(self) -> dict:
        """Explicitly apply the frozen lossless codec to a legacy or current state."""
        prior = self._snapshot
        blob, selection = select(self._state)
        if prior is not None and len(prior) <= len(blob):
            blob = prior
            selection = {**selection, "selected": "retained_snapshot"}
        self._snapshot = blob
        self._selection = selection
        return {**selection, "actual_snapshot_bytes": len(blob)}

    def save(self, path: str | Path) -> int:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        blob = self.snapshot()
        atomic_write_bytes(path, blob)
        return len(blob)

    def handoff(self, target_core: torch.nn.Module | None = None) -> StorageV1:
        """Restore independent mutable state; no host conversion or history replay."""
        core = self._state.core if target_core is None else target_core
        return type(self).from_snapshot(core, self.snapshot())

    def reader(self, ranker, reader_factory: Callable = EvidenceReader):
        if self.observations == 0:
            raise ValueError("Use read() for the empty-state result")
        key = (reader_factory, id(ranker))
        if self._reader_key != key:
            self._reader = reader_factory(self._state, ranker)
            self._reader_key = key
        return self._reader

    def read(self, query: str, ranker, *, reader_factory: Callable = EvidenceReader,
             topk: int = 16, budget: int = 2000, mode: str = "support"):
        if self.observations == 0:
            return None, "", {
                "slots": [], "source_tokens": 0, "state_mutated": False,
                "original_history_loaded": False,
            }
        return self.reader(ranker, reader_factory).select_evidence(
            query, topk=topk, budget=budget, mode=mode
        )

    def release_reader(self) -> None:
        self._reader = None
        self._reader_key = None

    def accounting(self) -> dict:
        """Count owned state/cache objects; models and shared tables are separate.

        This is live-object accounting, not process RSS, allocator reservation,
        or a peak-memory measurement. The cached serialized snapshot is charged.
        """
        blob = self.snapshot()
        mem = self._state.mem
        names = (
            "keys", "scales", "topics", "references", "anchor_slots",
            "innovation_anchors", "innovations", "innovation_scales", "exceptions",
        )
        fields = {name: getattr(mem, name) for name in names}
        fields["bindings"] = {
            k: v for k, v in vars(mem.bindings).items() if k != "owner"
        }
        fields.update(meta=self._state.meta, events=self._state.events)
        active = owned_bytes(fields)
        serialized_cache = owned_bytes(blob)
        reader = reader_ledger(self._reader) if self._reader is not None else {
            "reader_owned_cpu_and_tensor_bytes": 0,
            "decoded_reader_cache_present": False,
        }
        public_tables = {
            name: getattr(mem, name) for name in (
                "global_cdf", "predictions", "context_tables", "context_counts",
                "offsets", "cdf_offsets", "mean", "full_basis", "axis_scale"
            )
        }
        core_tensors = list(self._state.core.parameters()) + list(self._state.core.buffers())
        return {
            "observations": self.observations,
            "source_token_ids": sum(int(n) for n in mem.bindings.lengths),
            "serialized_persistent_bytes": len(blob),
            "active_state_owned_bytes": active,
            "serialized_snapshot_cache_owned_bytes": serialized_cache,
            **reader,
            "online_private_owned_bytes": active + serialized_cache + reader["reader_owned_cpu_and_tensor_bytes"],
            "shared_codec_runtime_owned_bytes": owned_bytes(public_tables),
            "shared_core_tensor_bytes": sum(t.numel() * t.element_size() for t in core_tensors),
            "shared_codec_file_bytes": asset_path(mem.cfg.model_path).stat().st_size + asset_path(mem.cfg.basis_path).stat().st_size,
            "new_weight_bytes": 0,
            "scope": "Live owned objects; excludes Python interpreter, allocator reserve, temporary peaks and host models",
            "snapshot_is_not_runtime_RAM": True,
        }
