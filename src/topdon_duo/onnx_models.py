"""Opt-in, checksum-verified visual models; no network access during capture."""

import argparse
import hashlib
import os
import re
import ssl
import sys
import tempfile
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .model_archives import extract_model

MODELS: dict[str, dict[str, Any]] = {
    "style-mosaic": {
        "factor": 1,
        "rgb": True,
        "task": "style",
        "pixel_range": 255,
        "url": "https://huggingface.co/onnxmodelzoo/mosaic-9/resolve/main/mosaic-9.onnx",
        "sha256": "fa646dedade881243f8d5a2ceb7de2b93675b21fc24f7482894ac4851a9a0a47",
        "fixed_input": (224, 224),
    },
    "style-candy": {
        "factor": 1,
        "rgb": True,
        "task": "style",
        "pixel_range": 255,
        "url": "https://huggingface.co/onnxmodelzoo/candy-9/resolve/main/candy-9.onnx",
        "sha256": "9d11a3529d1e547da6ae07201d93484dbab2ec0a3614535752c8f40f0fe2968a",
        "fixed_input": (224, 224),
    },
    "style-rain-princess": {
        "factor": 1,
        "rgb": True,
        "task": "style",
        "pixel_range": 255,
        "fixed_input": (224, 224),
        "url": "https://huggingface.co/onnxmodelzoo/rain-princess-9/resolve/main/rain-princess-9.onnx",
        "sha256": "4162912e6f75fedef6f810ae989b9e10d3d5d43308dab34b027c850cf255e152",
    },
    "style-udnie": {
        "factor": 1,
        "rgb": True,
        "task": "style",
        "pixel_range": 255,
        "fixed_input": (224, 224),
        "url": "https://huggingface.co/onnxmodelzoo/udnie-9/resolve/main/udnie-9.onnx",
        # Resolve and retain the publisher's Git LFS checksum at installation.
        "sha256": None,
    },
    "style-pointillism": {
        "factor": 1,
        "rgb": True,
        "task": "style",
        "pixel_range": 255,
        "fixed_input": (224, 224),
        # The upstream filename is spelled "pointilism".
        "url": "https://huggingface.co/onnxmodelzoo/pointilism-9/resolve/main/pointilism-9.onnx",
        "sha256": "5ee2b8d4d6bc60a777f54e0fe96a1b717360a004b79d56c67390d4a975b14d98",
    },
    "style-line-art": {
        "factor": 1,
        "rgb": True,
        "task": "style",
        "output_channels": 1,
        # Bounded working resolution for the dynamic Informative Drawings export.
        "fixed_input": (256, 256),
        "url": "https://huggingface.co/rocca/informative-drawings-line-art-onnx/resolve/main/model.onnx",
        "sha256": "1fef40b8f7126d827e30fbebccf95ae9b0b391795df926bf9366a821bad4f498",
    },
    "style-animegan-sketch": {
        "factor": 1,
        "rgb": True,
        "task": "style",
        "layout": "nhwc",
        "pixel_range": 2,
        "input_offset": -1,
        "fixed_input": (512, 512),
        "noncommercial": True,
        "url": "https://github.com/TachibanaYoshino/AnimeGANv3_Portrait_Inference/releases/download/1.0/AnimeGANv3_PortraitSketch_25.onnx",
        "sha256": "86de643d216a387ae94d6881fc4197e3427e9dca31a030386a1526f8077c55d8",
    },
    "realesr-general-x4v3": {
        "factor": 4,
        "rgb": True,
        "url": "https://huggingface.co/skillsafe-ai/realesr-general-x4v3/resolve/main/model.onnx",
        "sha256": "a946f7a9397021b9b6b7e71df3d2821b04cc09ff244423b7ca79cb191ce4a00e",
        "static_coreml": True,
        "input_name": "input",
    },
    "dncnn-25": {
        "factor": 1,
        "rgb": False,
        "sha256": None,
        "local_export": True,
        "static_coreml": True,
    },
    "ffdnet-gray": {
        "factor": 1,
        "rgb": False,
        "sha256": None,
        "local_export": True,
        "static_coreml": True,
    },
    "espcn": {
        "factor": 3,
        "rgb": False,
        "url": "https://huggingface.co/onnxmodelzoo/super-resolution-10/resolve/main/super-resolution-10.onnx",
        "sha256": "85f36ff88cc504a24af5e0602148bc56a8aa09a58eca8c0da2756f3e8186035e",
    },
    "mewzoom": {
        "factor": 4,
        "rgb": True,
        "url": "https://huggingface.co/andrewdalpino/MewZoom-V0-4X/resolve/main/model.onnx",
        "sha256": "bb8a67cf943a430ecd6600cc68258491dfea38122361b312e3fe47c259d6e465",
    },
    "mewzoom-v0-2x": {
        "factor": 2,
        "rgb": True,
        "url": "https://huggingface.co/andrewdalpino/MewZoom-V0-2X/resolve/main/model.onnx",
        "sha256": "a297702366f69fa919deeea14b47396c4091fd9e780080675ecfa3144aeadeba",
    },
    "mewzoom-v1-2x": {
        "factor": 2,
        "rgb": True,
        "url": "https://huggingface.co/andrewdalpino/MewZoom-V1-2X/resolve/main/model.onnx",
        # Fetch the publisher's SHA256 LFS pointer over verified HTTPS at install.
        "sha256": None,
    },
    "mewzoom-v1-4x": {
        "factor": 4,
        "rgb": True,
        "url": "https://huggingface.co/andrewdalpino/MewZoom-V1-4X/resolve/main/model.onnx",
        "sha256": None,
        "max_bytes": 120_000_000,
    },
}


def model_path(model: str, *, spec: dict[str, Any] | None = None) -> Path:
    """
    Locate a catalog or custom model in the shared host-specific cache.
    """
    spec = MODELS[model] if spec is None else spec
    override = os.environ.get("TOPDON_MODEL_DIR")
    if override:
        base = Path(override)
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Caches" / "topdon-duo" / "models"
    else:
        base = (
            Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "topdon-duo" / "models"
        )
    checksum = spec["sha256"]
    filename = f"{model}-{checksum[:12]}.onnx" if checksum else f"{model}.onnx"
    return base / filename


def installed_checksum(model: str, *, spec: dict[str, Any] | None = None) -> str:
    """
    Resolve a pinned checksum or the publisher checksum retained at installation.
    """
    spec = MODELS[model] if spec is None else spec
    checksum = spec["sha256"]
    if checksum is None:
        checksum = model_path(model, spec=spec).with_suffix(".sha256").read_text().strip()
    if not re.fullmatch(r"[0-9a-f]{64}", checksum):
        raise ValueError("Invalid model checksum metadata; reinstall " + model)
    return checksum


def verified_model(model: str, *, spec: dict[str, Any] | None = None) -> bytes:
    """
    Read cached weights only after validating their pinned checksum.
    """
    path = model_path(model, spec=spec)
    try:
        data = path.read_bytes()
        checksum = installed_checksum(model, spec=spec)
    except OSError as exc:
        advice = "reload the Custom node" if spec is not None else "run " + install_command(model)
        raise ValueError("Model not installed; " + advice) from exc
    if hashlib.sha256(data).hexdigest() != checksum:
        advice = "remove the cached file and reload the Custom node" if spec is not None else "reinstall with " + install_command(model)
        raise ValueError("Model checksum mismatch; " + advice)
    return data


def install_command(model):
    if MODELS[model].get("local_export"):
        return "uv run scripts/export_visual_denoisers.py " + model
    return "uv run topdon-duo-models " + model


def download_ssl_context():
    """Supplement framework Python's missing roots with macOS's system bundle.

    Retain secure SSL defaults and explicit SSL_CERT_FILE/SSL_CERT_DIR overrides.
    Do not change global SSL configuration or Linux's normal trust configuration.
    """
    context = ssl.create_default_context()
    system_bundle = Path("/etc/ssl/cert.pem")
    if (
        sys.platform == "darwin"
        and not os.environ.get("SSL_CERT_FILE")
        and not os.environ.get("SSL_CERT_DIR")
        and system_bundle.is_file()
    ):
        context.load_verify_locations(cafile=str(system_bundle))
    return context


def download_model(
    model: str,
    progress: Callable[[int, int], None] | None = None,
    *,
    spec: dict[str, Any] | None = None,
) -> Path:
    """
    Atomically install checksum-verified direct weights or one declared archive member.
    """
    custom = spec is not None
    spec = MODELS[model] if spec is None else spec
    path = model_path(model, spec=spec)
    if path.exists():
        try:
            verified_model(model, spec=spec if custom else None)
        except ValueError:
            pass  # Replace a corrupt cache only after the new download verifies.
        else:
            return path
    if spec.get("local_export"):
        raise ValueError("This model needs a one-time export; run " + install_command(model))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    checksum_temporary = None
    extracted = None
    try:
        context = download_ssl_context()
        checksum = spec["sha256"]
        maximum = spec.get("max_bytes", 70_000_000)
        if checksum is None:
            pointer_url = spec["url"].replace("/resolve/", "/raw/")
            with urllib.request.urlopen(pointer_url, timeout=30, context=context) as source:
                pointer = source.read(1025).decode("ascii")
            match = re.fullmatch(
                r"version https://git-lfs.github.com/spec/v1\noid sha256:([0-9a-f]{64})\nsize ([0-9]+)\n?",
                pointer,
            )
            if not match:
                raise ValueError(
                    "Invalid publisher model pointer; not downloading unverified weights"
                )
            checksum, advertised = match.group(1), int(match.group(2))
            if advertised > maximum:
                raise ValueError(
                    f"Publisher model exceeds the {maximum // 1_000_000} MB safety limit"
                )
        digest = hashlib.sha256()
        with (
            urllib.request.urlopen(spec["url"], timeout=30, context=context) as source,
            tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as target,
        ):
            temporary = Path(target.name)
            total = 0
            headers = getattr(source, "headers", {})
            try:
                length = int(headers.get("Content-Length", 0))
            except (TypeError, ValueError):
                length = 0
            while chunk := source.read(1024 * 1024):
                total += len(chunk)
                if total > maximum:
                    raise ValueError(
                        f"Model download exceeds the {maximum // 1_000_000} MB safety limit"
                    )
                digest.update(chunk)
                target.write(chunk)
                if progress is not None:
                    progress(total, length)
        if progress is not None:
            progress(total, total)
        if spec.get("archive_member"):
            if digest.hexdigest() != spec["archive_sha256"]:
                raise ValueError("Model archive checksum mismatch; not installed")
            extracted, model_checksum = extract_model(temporary, path.parent, spec["archive_member"], maximum)
            temporary.unlink()
            temporary = extracted
        else:
            model_checksum = digest.hexdigest()
        if model_checksum != checksum:
            raise ValueError("Model download checksum mismatch; not installed")
        if spec["sha256"] is None:
            with tempfile.NamedTemporaryFile(dir=path.parent, delete=False, mode="w") as target:
                checksum_temporary = Path(target.name)
                target.write(checksum + "\n")
        temporary.replace(path)
        if checksum_temporary is not None:
            checksum_temporary.replace(path.with_suffix(".sha256"))
        return path
    finally:
        if extracted is not None:
            extracted.unlink(missing_ok=True)
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        if checksum_temporary is not None:
            checksum_temporary.unlink(missing_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Install optional visual-only ONNX models")
    parser.add_argument("models", nargs="+", choices=tuple(MODELS))
    args = parser.parse_args(argv)
    try:
        for model in args.models:
            if MODELS[model].get("noncommercial"):
                print("Note: AnimeGANv3 weights are for non-commercial use only; see README for upstream terms.", flush=True)
            print(f"Installing/verifying {model}...", flush=True)
            print(f"{model}: {download_model(model)}")
    except (OSError, ValueError) as exc:
        print(f"Model installation failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
