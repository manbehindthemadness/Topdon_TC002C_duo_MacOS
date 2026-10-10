"""Install an optional Linux CUDA runtime into the current virtual environment."""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import sys


def install_command(uv: str, cuda: str, system_libraries: bool) -> list[str]:
    """Keep CUDA 12 installations below the release that switched PyPI to CUDA 13."""
    versions = {"12": ">=1.23,<1.27", "13": ">=1.27,<1.31"}
    extras = "" if system_libraries else "[cuda,cudnn]"
    return [
        uv, "pip", "install", "--python", sys.executable,
        f"onnxruntime-gpu{extras}{versions[cuda]}",
    ]


def main(argv: list[str] | None = None) -> int:
    """Resolve compatible wheels before replacing CPU ORT, then verify inference."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cuda", choices=("12", "13"), required=True,
                        help="CUDA runtime family supported by your NVIDIA driver")
    parser.add_argument("--system-libraries", action="store_true",
                        help="Use existing CUDA/cuDNN libraries instead of pip NVIDIA packages")
    parser.add_argument("--dry-run", action="store_true",
                        help="Check wheel availability without changing the environment")
    args = parser.parse_args(argv)
    if sys.platform != "linux" or platform.machine().lower() not in {
        "x86_64", "amd64", "aarch64", "arm64",
    }:
        print("CUDA setup supports Linux x86-64 and ARM64 only.", file=sys.stderr)
        return 2
    if sys.prefix == sys.base_prefix:
        print("Run this command with the project's .venv/bin/python.", file=sys.stderr)
        return 2
    uv = shutil.which("uv")
    if uv is None:
        print("Install uv and ensure it is on PATH before CUDA setup.", file=sys.stderr)
        return 2
    command = install_command(uv, args.cuda, args.system_libraries)
    # Missing ARM64 wheels and unsupported Python/glibc versions must leave the
    # working runtime intact. uv validates native wheel tags for this interpreter.
    if subprocess.run([*command, "--dry-run"], check=False).returncode:
        print("No compatible GPU package resolved; the environment is unchanged. "
              "See docs/nvidia-linux.md for vendor builds.", file=sys.stderr)
        return 1
    if args.dry_run:
        print("Package resolution succeeded. GPU inference has not been tested.")
        return 0
    removed = subprocess.run(
        [uv, "pip", "uninstall", "--python", sys.executable, "onnxruntime", "onnxruntime-gpu"],
        check=False,
    )
    if removed.returncode:
        return 1
    if subprocess.run(command, check=False).returncode:
        # The distributions share files: never leave a partially replaced import.
        restored = subprocess.run(
            [uv, "pip", "install", "--python", sys.executable,
             "--reinstall", "onnxruntime>=1.23"], check=False,
        )
        print("GPU installation failed. " + (
            "CPU runtime restored." if not restored.returncode
            else "CPU recovery failed; run uv sync to repair the environment."
        ), file=sys.stderr)
        return 1
    # A disposable process contains native errors and startup timeouts. Import
    # the new runtime there, never in this installer process.
    try:
        probe = subprocess.run(
            [sys.executable, "-c",
             ("import json; from topdon_duo.nvidia_acceleration import nvidia_acceleration; "
              "print(json.dumps(nvidia_acceleration()))")],
            capture_output=True, text=True, check=False, timeout=30,
        )
        result = json.loads(probe.stdout) if not probe.returncode else {}
    except (ValueError, OSError, subprocess.TimeoutExpired):
        result = {}
    if not isinstance(result, dict) or result.get("available") is not True:
        print("GPU inference unavailable; viewer CPU fallback remains enabled. "
              "Run .venv/bin/python -m topdon_duo.nvidia_acceleration for details.",
              file=sys.stderr)
        return 1
    print("CUDA inference verified. Use .venv/bin/ executables or uv run --no-sync; "
          "uv sync restores the locked CPU runtime.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
