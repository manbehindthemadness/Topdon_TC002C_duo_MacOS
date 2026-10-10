"""
Exercise shared custom downloads with bounded fake network responses.
"""

import hashlib
import io
import tarfile
from pathlib import Path
from threading import Event
from time import monotonic, sleep
from typing import Any
from unittest.mock import Mock

import pytest

from topdon_duo import model_downloads, onnx_models
from topdon_duo.custom_nodes import models
from topdon_duo.custom_nodes.models import ModelResolver, model_source


def source(data: bytes) -> dict[str, Any]:
    """
    Pin an independently supplied fake model payload.
    """
    return {"name": "Test model", "url": "https://example.com/model.onnx",
            "sha256": hashlib.sha256(data).hexdigest()}


@pytest.mark.parametrize("change", [
    {"url": "http" + "://example.com/model"}, {"sha256": "bad"}, {"max_bytes": True},
    {"max_bytes": 512000001}, {"archive_member": "../model"},
    {"archive_member": "model", "archive_sha256": "bad"}, {"extra": 1},
])
def test_invalid_sources_never_request_network(change: dict[str, Any]) -> None:
    """
    Reject unpinned, oversized or unsafe sources before queuing a job.
    """
    with pytest.raises(ValueError):
        model_source(source(b"weights") | change)


def test_direct_download_reuses_cache_and_rejects_corruption(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """
    Verify direct custom weights without extending the built-in catalog.
    """
    monkeypatch.setenv("TOPDON_MODEL_DIR", str(tmp_path))
    data = b"custom weights"
    identity, spec = model_source(source(data))
    opener = Mock(side_effect=lambda *args, **kwargs: io.BytesIO(data))
    monkeypatch.setattr(onnx_models.urllib.request, "urlopen", opener)
    path = onnx_models.download_model(identity, spec=spec)
    assert onnx_models.download_model(identity, spec=spec) == path
    assert opener.call_count == 1
    assert identity not in onnx_models.MODELS
    resolver = ModelResolver()
    assert resolver.resolve(source(data)) == path
    path.write_bytes(b"corrupt weights")
    with pytest.raises(ValueError, match="checksum mismatch"):
        resolver.resolve(source(data))


def archive_bytes(data: bytes, kind: Any = tarfile.REGTYPE) -> bytes:
    """
    Construct a tiny archive with one model and an unrelated traversal entry.
    """
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        unrelated = tarfile.TarInfo("../outside")
        unrelated.size = 1
        archive.addfile(unrelated, io.BytesIO(b"x"))
        member = tarfile.TarInfo("saved_model/model.onnx")
        member.type = kind
        member.size = len(data) if kind == tarfile.REGTYPE else 0
        member.linkname = "../outside" if kind != tarfile.REGTYPE else ""
        archive.addfile(member, io.BytesIO(data) if member.size else None)
    return buffer.getvalue()


@pytest.mark.parametrize("failure", ["", "archive_hash", "model_hash", "member", "symlink", "size"])
def test_archive_installs_only_verified_regular_member_atomically(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, failure: str
) -> None:
    """
    Require both checksums and never unpack links or unrelated archive paths.
    """
    cache = tmp_path / "cache"
    monkeypatch.setenv("TOPDON_MODEL_DIR", str(cache))
    data = b"model weights"
    archive = archive_bytes(data, tarfile.SYMTYPE if failure == "symlink" else tarfile.REGTYPE)
    descriptor = source(data) | {
        "archive_member": "saved_model/model.onnx",
        "archive_sha256": hashlib.sha256(archive).hexdigest(),
    }
    if failure == "archive_hash":
        descriptor["archive_sha256"] = "0" * 64
    elif failure == "model_hash":
        descriptor["sha256"] = "0" * 64
    elif failure == "member":
        descriptor["archive_member"] = "missing.onnx"
    elif failure == "size":
        descriptor["max_bytes"] = 1
    identity, spec = model_source(descriptor)
    monkeypatch.setattr(onnx_models.urllib.request, "urlopen", lambda *args, **kwargs: io.BytesIO(archive))
    if failure:
        with pytest.raises(ValueError):
            onnx_models.download_model(identity, spec=spec)
        assert list(cache.iterdir()) == []
    else:
        path = onnx_models.download_model(identity, spec=spec)
        assert path.read_bytes() == data
        assert list(cache.iterdir()) == [path]
    assert not (tmp_path / "outside").exists()


def test_shared_job_progress_deduplication_and_offline_readiness(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """
    Multiple custom packages share one job and obtain verified cached paths.
    """
    monkeypatch.setenv("TOPDON_MODEL_DIR", str(tmp_path))
    manager = model_downloads.ModelDownloads()
    # noinspection PyUnresolvedReferences
    monkeypatch.setattr(models, "MODEL_DOWNLOADS", manager)
    data = b"downloaded weights"
    started, release, finished = Event(), Event(), Event()

    def install(model: str, progress: Any, *, spec: dict[str, Any]) -> None:
        """
        Hold installation long enough to exercise two concurrent consumers.
        """
        progress(1_000_000, 2_000_000)
        started.set()
        assert release.wait(2)
        onnx_models.model_path(model, spec=spec).write_bytes(data)
        finished.set()

    installer = Mock(side_effect=install)
    # noinspection PyUnresolvedReferences
    monkeypatch.setattr(model_downloads, "download_model", installer)
    first, second = ModelResolver(), ModelResolver()
    try:
        assert first.resolve(source(data)) is None
        assert started.wait(2)
        assert second.resolve(source(data)) is None
        assert "Test model: 1.0 MB / 2.0 MB" in manager.status()
        assert installer.call_count == 1
    finally:
        release.set()
    assert finished.wait(2)
    identity, _ = model_source(source(data))
    assert onnx_models.model_path(identity, spec=source(data)).read_bytes() == data
    deadline = monotonic() + 2
    path = None
    while path is None and monotonic() < deadline:
        path = first.resolve(source(data))
        sleep(0.001)
    assert path is not None and second.resolve(source(data)) == path


def test_retry_once_per_package_and_builtin_catalog_access(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Poll pending jobs without repeated retries and retain built-in name support.
    """
    request = Mock(return_value=False)
    # noinspection PyUnresolvedReferences
    monkeypatch.setattr(models.MODEL_DOWNLOADS, "request", request)
    resolver = ModelResolver()
    assert resolver.resolve("espcn") is None
    assert resolver.resolve("espcn") is None
    assert [call.kwargs["retry"] for call in request.call_args_list] == [True, False]
    assert all("spec" not in call.kwargs for call in request.call_args_list)
    assert ModelResolver().resolve("espcn") is None
    assert request.call_args.kwargs["retry"]
