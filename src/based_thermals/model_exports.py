"""
Prepare supported denoisers in an isolated uv subprocess, outside acquisition and inference.
"""

import fcntl
import os
import signal
import subprocess
import sys
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from importlib.resources import as_file, files
from pathlib import Path

from .onnx_models import model_path, verified_model

EXPORT_MODELS = ("dncnn-25", "ffdnet-gray")
EXPORT_TIMEOUT = 900


@contextmanager
def _export_script() -> Iterator[Path]:
    """
    Locate the bundled converter in installed wheels or the existing source checkout.
    """
    resource = files("based_thermals").joinpath("resources", "export_visual_denoisers.py")
    if resource.is_file():
        with as_file(resource) as path:
            yield path
        return
    path = Path(__file__).resolve().parents[2] / "scripts" / "export_visual_denoisers.py"
    if not path.is_file():
        raise ValueError("The installed application is missing its denoiser converter")
    yield path


def _ready(model: str) -> bool:
    """
    Reuse only a model whose cached bytes match its checksum.
    """
    try:
        verified_model(model)
    except ValueError:
        return False
    return True


def run_export(command: list[str], environment: dict[str, str], directory: Path) -> None:
    """
    Bound dependency setup and conversion, terminating the whole subprocess group on timeout.
    """
    process = subprocess.Popen(
        command, env=environment, cwd=directory, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True, start_new_session=True,
    )
    try:
        output, _ = process.communicate(timeout=EXPORT_TIMEOUT)
    except subprocess.TimeoutExpired as exc:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.communicate()
        raise ValueError(f"Model preparation timed out after {EXPORT_TIMEOUT} seconds") from exc
    if process.returncode:
        detail = (output or "Converter exited without a diagnostic").strip()[-2000:]
        raise ValueError(detail)


def prepare_model(model: str, progress: Callable[[str], None] | None = None) -> Path:
    """
    Coordinate one-time conversion across viewers and reuse verified models offline.
    """
    if model not in EXPORT_MODELS:
        raise ValueError(f"Automatic conversion is not supported for {model}")
    path = model_path(model)
    path.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + EXPORT_TIMEOUT
    with (path.parent / f".{model}.export.lock").open("a") as lock:
        while True:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise ValueError("Timed out waiting for another viewer to prepare this model")
                if progress is not None:
                    progress("Waiting for another viewer's model preparation…")
                time.sleep(0.1)
        try:
            if _ready(model):
                return path
            from .runtime_setup.install import find_uv

            uv = find_uv()
            if uv is None:
                raise ValueError("Automatic model preparation needs uv on PATH or beside Python")
            environment = os.environ.copy()
            for name in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV", "BASED_THERMALS_ORT_PATH"):
                environment.pop(name, None)
            environment["BASED_THERMALS_MODEL_DIR"] = str(path.parent.resolve())
            environment["BASED_THERMALS_EXPORT_SOURCE_ROOT"] = str(Path(__file__).resolve().parents[1])
            if progress is not None:
                progress("Preparing conversion dependencies and model (first use)…")
            with _export_script() as script:
                command = [uv, "run", "--no-project", "--no-python-downloads", "--python",
                           sys.executable, "--script", str(script), model]
                run_export(command, environment, path.parent)
            verified_model(model)
            return path
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)
