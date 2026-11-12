"""Download shared assets and optional external models for the local runtime."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys
import tarfile
import urllib.request

RELEASE_URL = (
    "https://github.com/zOUMINGXI/MIGRO/releases/download/"
    "runtime-assets-20261112/migro-runtime-assets-20261112.tar.gz"
)
MODELS = {
    "qwen": ("Qwen/Qwen3-4B-Instruct-2507", "Qwen3-4B-Instruct-2507"),
    "mistral": ("mistralai/Mistral-7B-Instruct-v0.3", "Mistral-7B-Instruct-v0.3"),
    "phi": ("microsoft/Phi-3-mini-4k-instruct", "Phi-3-mini-4k-instruct"),
    "reranker": ("cross-encoder/ms-marco-MiniLM-L6-v2", "ms-marco-MiniLM-L6-v2"),
}


def install_assets(directory: Path, url: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=120) as response:
        with tarfile.open(fileobj=response, mode="r|gz") as archive:
            for member in archive:
                if member.isdir():
                    continue
                relative = PurePosixPath(member.name)
                if (
                    not member.isfile()
                    or relative.is_absolute()
                    or ".." in relative.parts
                    or relative.parts[0] != "migro-assets"
                ):
                    raise ValueError(f"Unexpected archive member: {member.name}")
                target = directory.joinpath(*relative.parts[1:])
                target.parent.mkdir(parents=True, exist_ok=True)
                temporary = target.with_name(target.name + ".partial")
                with archive.extractfile(member) as source, temporary.open("wb") as output:
                    while block := source.read(1024 * 1024):
                        output.write(block)
                temporary.replace(target)
    manifest = json.loads((directory / "ASSETS.json").read_text())
    for entry in manifest["files"]:
        asset = directory / entry["path"]
        if asset.stat().st_size != entry["bytes"]:
            raise ValueError(f"Incomplete asset: {entry['path']}")
    print(f"Shared assets installed in {directory}")


def download_models(directory: Path, names: list[str]) -> None:
    from huggingface_hub import snapshot_download

    for name in dict.fromkeys(names):
        model_id, folder = MODELS[name]
        snapshot_download(
            model_id,
            local_dir=directory / "models" / folder,
            allow_patterns=[
                "*.json", "*.safetensors", "*.model", "vocab.txt", "merges.txt",
                "tokenizer.*", "LICENSE*", "NOTICE*", "README.md",
            ],
        )
        print(f"Model available: {model_id}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets-dir", type=Path, required=True)
    parser.add_argument("--url", default=RELEASE_URL)
    parser.add_argument("--skip-assets", action="store_true")
    parser.add_argument("--models", nargs="*", choices=sorted(MODELS), default=[])
    args = parser.parse_args()
    directory = args.assets_dir.expanduser().resolve()
    if not args.skip_assets:
        install_assets(directory, args.url)
    if args.models:
        download_models(directory, args.models)
    manifest = json.loads((directory / "ASSETS.json").read_text())
    for entry in manifest["files"]:
        if (directory / entry["path"]).stat().st_size != entry["bytes"]:
            raise ValueError(f"Incomplete asset: {entry['path']}")
    environment = dict(os.environ, MIGRO_ASSETS=str(directory))
    subprocess.run(
        [sys.executable, "-m", "migro.backend.build_native"],
        env=environment,
        check=True,
    )
    print("Runtime assets and native kernels are ready.")


if __name__ == "__main__":
    main()
