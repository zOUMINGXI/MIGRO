"""Build the original exact coding kernels without model weights."""
import argparse, os, subprocess
from pathlib import Path
from .paths import native_dir

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=native_dir())
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    source = Path(__file__).with_name("native")
    compiler = os.environ.get("CXX", "c++")
    for filename, output, shared in [
        ("rd_ans_native.cpp", "libsemantic_ans.so", True),
        ("rd_anchor_native.cpp", "libanchors.so", True),
        ("causal_content.cpp", "causal_content", False),
    ]:
        command = [compiler, "-O3", "-std=c++17"]
        if shared:
            command += ["-shared", "-fPIC"]
        subprocess.run(command + [str(source / filename), "-o", str(args.output / output)], check=True)
        print(args.output / output)

if __name__ == "__main__":
    main()
