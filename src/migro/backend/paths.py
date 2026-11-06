"""Local resource paths shared by the packaged research backend."""
import os
from pathlib import Path

def asset_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return Path(os.environ.get("MIGRO_ASSETS", "artifacts")).expanduser().resolve() / path

def native_dir() -> Path:
    return Path(os.environ.get("MIGRO_NATIVE_DIR", str(asset_path("native")))).expanduser()

def model_path(name: str, default: str) -> str:
    return os.environ.get("MIGRO_" + name + "_MODEL", str(asset_path("models/" + default)))
