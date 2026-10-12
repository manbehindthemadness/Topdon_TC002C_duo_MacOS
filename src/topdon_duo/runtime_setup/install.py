"""
Stage optional native runtimes without replacing uv-managed environment packages.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import sys
from email.parser import Parser
from pathlib import Path

LOG = logging.getLogger(__name__)
REQUIREMENTS = {
    "cpu": "onnxruntime>=1.23,<1.31",
    "coreml": "onnxruntime>=1.23,<1.31",
    "cuda12": "onnxruntime-gpu>=1.23,<1.27",
    "cuda13": "onnxruntime-gpu>=1.27,<1.31",
    # Newer ARM64 PyPI wheels can omit Orin kernels despite matching Python tags.
    # Keep the physically verified Tegra234 build isolated from generic CUDA hosts.
    "cuda13-orin": "onnxruntime-gpu==1.28.0.dev20260722004",
}
ORIN_INDEX = (
    "https://aiinfra.pkgs.visualstudio.com/PublicPackages/"
    "_packaging/ort-cuda-13-nightly/pypi/simple/"
)


def find_uv() -> str | None:
    """
    Support both PATH-installed uv and the README's environment-local Ubuntu bootstrap.
    """
    local = Path(sys.executable).parent / "uv"
    executable = shutil.which("uv")
    return executable or (str(local) if local.is_file() else None)


def install(
    uv: str, path: Path, requirements: list[str], *, dependencies: bool = False,
    index_url: str | None = None,
) -> bool:
    """
    Resolve wheels for the running interpreter and contain network/installer failures.
    """
    command = [uv, "pip", "install", "--python", sys.executable, "--target", str(path)]
    if not dependencies:
        command.append("--no-deps")
    if index_url is not None:
        command.extend(["--index-url", index_url])
    command.extend(requirements)
    LOG.info("Setting up ONNX Runtime: %s", ", ".join(requirements))
    try:
        completed = subprocess.run(command, check=False, capture_output=True, text=True, timeout=600)
        if completed.returncode:
            LOG.warning("ONNX Runtime installation failed: %s", completed.stderr[-4000:])
        return completed.returncode == 0
    except (OSError, subprocess.TimeoutExpired) as exc:
        LOG.warning("ONNX Runtime installation unavailable: %s", exc)
        return False


def cuda_dependencies(path: Path) -> list[str]:
    """
    Use the staged wheel's CUDA/cuDNN extra requirements, leaving base Python packages alone.
    """
    metadata_files = list(path.glob("onnxruntime_gpu-*.dist-info/METADATA"))
    if len(metadata_files) != 1:
        raise ValueError("Missing GPU wheel metadata")
    metadata = Parser().parsestr(metadata_files[0].read_text(encoding="utf-8"))
    requirements = []
    for requirement in metadata.get_all("Requires-Dist", []):
        package, _, marker = requirement.partition(";")
        if package.strip().startswith("nvidia-") and any(
            f'extra == "{extra}"' in marker or f"extra == '{extra}'" in marker
            for extra in ("cuda", "cudnn")
        ):
            requirements.append(package.strip())
    if not requirements:
        raise ValueError("GPU wheel does not declare CUDA/cuDNN installation extras")
    return requirements
