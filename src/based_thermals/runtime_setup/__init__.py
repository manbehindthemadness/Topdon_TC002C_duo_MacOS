"""
Provision and activate verified ONNX runtimes before acquisition or worker startup.
"""

from __future__ import annotations

import json
import logging
import os
import platform
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, cast

from ..apple_acceleration import apple_acceleration as apple_acceleration
from ..nvidia_acceleration import nvidia_acceleration as nvidia_acceleration
from .install import ORIN_INDEX, REQUIREMENTS, cuda_dependencies
from .install import find_uv as find_uv
from .install import install as install_packages
from .probes import environment
from .probes import query as query

LOG = logging.getLogger(__name__)
RETRY_SECONDS = 300

__all__ = [
    "activate", "apple_acceleration", "backend", "cache_root", "cached_path", "candidates",
    "ensure_runtime", "find_uv", "install_packages", "nvidia_acceleration", "prepare", "query",
    "read_state", "write_state",
]


def cache_root() -> Path:
    """
    Keep native runtime packages separate from both uv's environment and model downloads.
    """
    override = os.environ.get("BASED_THERMALS_RUNTIME_DIR")
    if override:
        return Path(override).expanduser().resolve()
    base = (
        Path.home() / "Library" / "Caches" if sys.platform == "darwin"
        else Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    )
    return base / "topdon-duo" / "runtimes"


def orin_host() -> bool:
    """
    Identify Tegra234 without treating other ARM64 NVIDIA hosts as Jetson Orin.
    """
    try:
        return b"nvidia,tegra234" in Path("/proc/device-tree/compatible").read_bytes().split(b"\0")
    except OSError:
        return False


def candidates() -> list[str]:
    """
    Select supported host backends from CUDA driver capabilities rather than product names.
    """
    if sys.platform == "darwin":
        return ["coreml", "cpu"] if query("metal").get("available") is True else ["cpu"]
    if sys.platform == "linux" and platform.machine().lower() in {
        "x86_64", "amd64", "aarch64", "arm64",
    }:
        version = query("driver").get("cuda", 0)
        if version >= 13000:
            if platform.machine().lower() in {"aarch64", "arm64"} and orin_host():
                return ["cuda13-orin", "cuda13", "cuda12", "cpu"]
            return ["cuda13", "cuda12", "cpu"]
        if version >= 12000:
            return ["cuda12", "cpu"]
    return ["cpu"]


def backend(selection: str) -> str:
    """
    Map a package family to the existing inference backend name.
    """
    return "cuda" if selection.startswith("cuda") else selection


def activate(path: Path) -> None:
    """
    Share the same runtime with current, subprocess, and multiprocessing inference workers.
    """
    previous = os.environ.get("BASED_THERMALS_ORT_PATH")
    if previous and previous in sys.path:
        sys.path.remove(previous)
    selected = str(path)
    if selected in sys.path:
        sys.path.remove(selected)
    sys.path.insert(0, selected)
    env = environment(path)
    os.environ["PYTHONPATH"] = env["PYTHONPATH"]
    os.environ["BASED_THERMALS_ORT_PATH"] = selected
    apple_acceleration.cache_clear()
    nvidia_acceleration.cache_clear()


def read_state(path: Path) -> dict[str, Any]:
    """
    Treat missing or malformed setup state as a fresh install, without executing its contents.
    """
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
        return result if isinstance(result, dict) else {}
    except (OSError, ValueError):
        return {}


def cached_path(root: Path, state: dict[str, Any]) -> Path | None:
    """
    Accept only a complete runtime generation directly under its application-owned cache.
    """
    name = state.get("ready")
    if not isinstance(name, str):
        return None
    if not name.startswith("runtime-") or Path(name).name != name:
        return None
    path = root / cast(str, name)
    if path.is_symlink() or not (path / "onnxruntime" / "__init__.py").is_file():
        return None
    return path


def write_state(path: Path, state: dict[str, Any]) -> None:
    """
    Publish startup state atomically while holding the setup lock.
    """
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(state), encoding="utf-8")
    temporary.replace(path)


def prepare(selection: str, *, retry: bool) -> bool:
    """
    Reuse or stage a verified runtime, coordinating launches without touching the project venv.
    """
    import fcntl

    if query(backend(selection)).get("available"):
        return True
    tag = f"v1-{sys.implementation.cache_tag}-{sys.platform}-{platform.machine()}-{selection}"
    root = cache_root() / tag
    root.mkdir(parents=True, exist_ok=True)
    with (root / "setup.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            LOG.warning("Another process is setting up %s; using existing runtime for this launch", selection)
            return False
        for abandoned in root.glob("staging-*"):
            if abandoned.is_dir() and not abandoned.is_symlink():
                shutil.rmtree(abandoned, ignore_errors=True)
        state_file = root / "state.json"
        state = read_state(state_file)
        cached = cached_path(root, state)
        if cached is not None and query(backend(selection), cached).get("available") is True:
            activate(cached)
            LOG.info("Using cached ONNX Runtime: %s", selection)
            return True
        failed = state.get("failed_at")
        if (
            not retry and isinstance(failed, (int, float)) and not isinstance(failed, bool)
            and 0 <= time.time() - failed < RETRY_SECONDS
        ):
            LOG.warning("Recent %s setup failure; retry deferred (run runtime_setup --retry)", selection)
            return False
        uv = find_uv()
        if uv is None:
            LOG.warning("ONNX Runtime setup needs uv on PATH or beside the project interpreter")
            return False
        stage = Path(tempfile.mkdtemp(prefix="staging-", dir=root))
        ready = False
        try:
            options = {"index_url": ORIN_INDEX} if selection == "cuda13-orin" else {}
            if not install_packages(uv, stage, [REQUIREMENTS[selection]], **options):
                return False
            result = query(backend(selection), stage)
            if (
                result.get("available") is not True and selection.startswith("cuda")
                and selection != "cuda13-orin"
            ):
                LOG.info("System CUDA libraries unavailable: %s", result.get("reason"))
                requirements = cuda_dependencies(stage)
                if not install_packages(uv, stage, requirements, dependencies=True):
                    return False
                result = query("cuda", stage)
            if not result.get("available"):
                LOG.warning("%s runtime verification failed: %s", selection, result.get("reason"))
                return False
            committed = root / stage.name.replace("staging-", "runtime-", 1)
            stage.rename(committed)
            stage = committed
            write_state(state_file, {"ready": stage.name})
            activate(stage)
            ready = True
            LOG.info("ONNX Runtime setup completed: %s", result.get("reason"))
            return True
        finally:
            if not ready:
                shutil.rmtree(stage, ignore_errors=True)
                write_state(state_file, {**state, "failed_at": time.time()})


def ensure_runtime(*, retry: bool = False) -> None:
    """
    Repair supported runtimes before camera access, with offline and native failure fallback.
    """
    if os.environ.get("BASED_THERMALS_RUNTIME_SETUP") == "0":
        LOG.info("Automatic ONNX Runtime setup disabled")
        return
    if sys.platform not in {"darwin", "linux"}:
        return
    if "onnxruntime" in sys.modules:
        LOG.warning("ONNX Runtime already imported; restart the application to run runtime setup")
        return
    for selection in candidates():
        try:
            if prepare(selection, retry=retry):
                return
        except (OSError, ValueError) as exc:
            LOG.warning("%s runtime setup unavailable: %s", selection, exc)
    LOG.warning("No runtime passed startup verification; continuing with the installed CPU fallback")
