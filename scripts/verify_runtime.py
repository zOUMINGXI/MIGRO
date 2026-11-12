"""Check a downloaded MIGRO bundle using the installed public runtime."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory


def verify(directory: Path, device: str) -> dict:
    os.environ["MIGRO_ASSETS"] = str(directory)
    import torch
    import transformers
    from migro.memory import MemoryStore, load_core
    from migro.backend.rd_boundary_interface import BoundaryAdapter
    from migro.backend.rd_third_host import ResidualInterface

    torch.set_num_threads(4)
    torch.set_grad_enabled(False)
    manifest = json.loads((directory / "ASSETS.json").read_text())
    for entry in manifest["files"]:
        if (directory / entry["path"]).stat().st_size != entry["bytes"]:
            raise ValueError(f"Asset size mismatch: {entry['path']}")
    core = load_core(device)
    rows = [
        {
            "speaker": "Ada", "role": "observed_human", "body": body,
            "session": "example", "date": "1 January, 2026", "dialogue_id": f"example:{i}",
        }
        for i, body in enumerate([
            "I live in Kyoto.", "My favorite color is blue.", "I now live in Osaka.",
        ])
    ]
    source = MemoryStore.create(core)
    source.observe(rows[:2])
    before = source.state.serialize()
    with TemporaryDirectory() as temporary:
        path = Path(temporary) / "memory.bin"
        source.save(path)
        receiver = MemoryStore.load(core, path).handoff()
    old_ids = [receiver.state.mem.read_slot(i, False).clone() for i in range(2)]
    receiver.observe(rows[2:])
    assert source.state.serialize() == before
    assert receiver.observations == 3
    for i, expected in enumerate(old_ids):
        assert torch.equal(expected, receiver.state.mem.read_slot(i, False))
    receiver.compact()
    restored = MemoryStore.from_snapshot(core, receiver.snapshot())
    assert restored.state.meta == receiver.state.meta
    for i in range(3):
        assert torch.equal(
            restored.state.mem.read_slot(i, False),
            receiver.state.mem.read_slot(i, False),
        )

    interfaces = directory / "research/decomem_v1_20261028/frozen_interfaces"
    for filename, adapter in [
        ("Phi_boundary.pt", BoundaryAdapter(3072)),
        ("Mistral_cal64.pt", ResidualInterface(4096)),
    ]:
        checkpoint = torch.load(interfaces / filename, map_location="cpu", weights_only=True)
        assert set(checkpoint) == {"adapter", "interface_scope"}
        adapter.load_state_dict(checkpoint["adapter"], strict=True)

    return {
        "completed": True,
        "device": device,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "payload_files": len(manifest["files"]),
        "write_save_load_handoff_append_compact_restore": True,
        "earlier_bindings_preserved": True,
        "interface_checkpoints_load": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets-dir", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()
    print(json.dumps(verify(args.assets_dir.expanduser().resolve(), args.device), indent=2))


if __name__ == "__main__":
    main()
