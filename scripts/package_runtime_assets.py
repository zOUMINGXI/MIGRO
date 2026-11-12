"""Prepare the redistributable shared runtime bundle."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import tarfile

import numpy as np
import torch


def tensor_mapping(value):
    return {
        name: tensor.detach().cpu()
        for name, tensor in value.items()
        if torch.is_tensor(tensor)
    }


def write_adapter(source: Path, destination: Path) -> dict:
    original = torch.load(source, map_location="cpu", weights_only=True)
    adapter = original["adapter"]
    cleaned = {"adapter": tensor_mapping(adapter)}
    scope = original.get(
        "interface_scope", original.get("config", {}).get("interface_scope", "memory")
    )
    cleaned["interface_scope"] = scope
    torch.save(cleaned, destination)
    restored = torch.load(destination, map_location="cpu", weights_only=True)
    assert set(restored["adapter"]) == set(adapter)
    for name, before in adapter.items():
        after = restored["adapter"][name]
        assert before.dtype == after.dtype and before.shape == after.shape
        assert torch.equal(before.cpu(), after)
    return {
        "tensor_count": len(cleaned["adapter"]),
        "parameters": sum(value.numel() for value in cleaned["adapter"].values()),
        "interface_scope": scope,
        "retained_fields": sorted(cleaned),
        "weights_equal": True,
    }


def prepare(source: Path, destination: Path, notices: Path) -> dict:
    old_manifest = json.loads((source / "ASSETS.json").read_text())
    destination.mkdir(parents=True, exist_ok=True)
    adapter_checks = {}
    files = []
    for entry in old_manifest["files"]:
        relative = Path(entry["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Invalid asset path: {relative}")
        original = source / relative
        output = destination / relative
        output.parent.mkdir(parents=True, exist_ok=True)
        if relative.parent.name == "frozen_interfaces":
            adapter_checks[relative.name] = write_adapter(original, output)
        elif relative.suffix == ".pt":
            state = torch.load(original, map_location="cpu", weights_only=True)
            if not all(torch.is_tensor(value) for value in state.values()):
                raise ValueError(f"Unexpected metadata in shared weights: {relative}")
            shutil.copyfile(original, output)
        elif relative.suffix == ".npz":
            with np.load(original, allow_pickle=False) as arrays:
                if any(arrays[name].dtype.kind == "O" for name in arrays.files):
                    raise ValueError(f"Object metadata in coding tables: {relative}")
            shutil.copyfile(original, output)
        elif relative.suffix == ".json":
            document = json.loads(original.read_text())
            # Local cache markers are unnecessary for loading the exported tokenizer.
            document.pop("is_local", None)
            document.pop("_name_or_path", None)
            output.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n")
        else:
            shutil.copyfile(original, output)
        files.append({"path": relative.as_posix(), "bytes": output.stat().st_size})

    for name in ["LICENSE", "NOTICE", "THIRD_PARTY_NOTICES.md", "PROVENANCE.json"]:
        shutil.copyfile(notices / name, destination / name)
    shutil.copytree(notices / "licenses", destination / "licenses", dirs_exist_ok=True)
    manifest = {
        "name": "MIGRO shared runtime assets",
        "date": "2026-11-12",
        "license": "Apache-2.0 for MIGRO components; upstream components retain their licenses",
        "files": files,
        "provenance": "PROVENANCE.json",
        "third_party_notices": "THIRD_PARTY_NOTICES.md",
        "excludes": [
            "backbone and reranker model weights",
            "benchmark histories and answers",
            "per-user memory states",
            "training configuration and optimizer state",
            "native binaries",
        ],
    }
    (destination / "ASSETS.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    )
    return {"payload_files": len(files), "adapters": adapter_checks}


def write_archive(directory: Path, archive: Path) -> None:
    archive.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "w:gz") as output:
        output.add(directory, arcname="migro-assets")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--notices", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    args = parser.parse_args()
    report = prepare(args.source, args.destination, args.notices)
    write_archive(args.destination, args.archive)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
