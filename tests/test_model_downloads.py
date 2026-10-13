import hashlib
from collections.abc import Callable
from pathlib import Path
from threading import Event
from time import monotonic, sleep

import numpy as np
import pytest

from based_thermals import model_downloads, onnx_models
from based_thermals.onnx_upsampling import ONNXUpsampler


def wait_finished(manager: model_downloads.ModelDownloads, model: str) -> None:
    """
    Wait for a bounded fake background job to finish.
    """
    deadline = monotonic() + 2
    while monotonic() < deadline:
        with manager.lock:
            if manager.jobs[model]["state"] != "pending":
                return
        sleep(0.005)
    pytest.fail("background installation did not finish")


def test_background_download_deduplicates_branches_and_reuses_offline_cache(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """
    Share background installation across branches and reuse the current model cache.
    """
    monkeypatch.setenv("BASED_THERMALS_MODEL_DIR", str(tmp_path))
    started, release = Event(), Event()
    calls = []

    def install(model: str, progress: Callable[[int, int], None]) -> None:
        """
        Hold a fake weight download until the test releases it.
        """
        calls.append(model)
        progress(1_000_000, 2_000_000)
        started.set()
        assert release.wait(2)
        onnx_models.model_path(model).write_bytes(b"test weights")

    # noinspection PyUnresolvedReferences
    monkeypatch.setattr(model_downloads, "download_model", install)
    manager = model_downloads.ModelDownloads()
    try:
        assert not manager.request("style-candy")
        assert started.wait(2)
        assert not manager.request("style-candy", retry=True)
        assert calls == ["style-candy"]
        assert "1.0 MB / 2.0 MB" in manager.status()
    finally:
        release.set()
    wait_finished(manager, "style-candy")
    assert manager.request("style-candy")
    assert manager.status() == ""
    assert calls == ["style-candy"]


def test_failed_download_does_not_retry_every_frame_but_can_retry(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """
    Keep failed downloads stable until an explicit retry is requested.
    """
    monkeypatch.setenv("BASED_THERMALS_MODEL_DIR", str(tmp_path))
    calls = []

    def install(model: str, progress: Callable[[int, int], None]) -> None:
        """
        Fail a fake download after reporting progress.
        """
        calls.append(model)
        progress(0, 0)
        raise OSError("offline")

    # noinspection PyUnresolvedReferences
    monkeypatch.setattr(model_downloads, "download_model", install)
    manager = model_downloads.ModelDownloads()
    assert not manager.request("style-candy")
    wait_finished(manager, "style-candy")
    for _ in range(3):
        with pytest.raises(ValueError, match="offline"):
            manager.request("style-candy")
    assert calls == ["style-candy"]
    assert not manager.request("style-candy", retry=True)
    wait_finished(manager, "style-candy")
    assert calls == ["style-candy", "style-candy"]


@pytest.mark.parametrize("model", ["dncnn-25", "ffdnet-gray"])
def test_local_export_is_prepared_once_in_background(
    model: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """
    Share automatic preparation across branches without waiting or importing converter libraries.
    """
    monkeypatch.setenv("BASED_THERMALS_MODEL_DIR", str(tmp_path))
    started, release = Event(), Event()
    calls: list[str] = []

    def prepare(name: str, progress: Callable[[str], None]) -> Path:
        """
        Hold a fake conversion while branches continue returning pending status.
        """
        calls.append(name)
        progress("Converting and verifying")
        started.set()
        assert release.wait(2)
        path = onnx_models.model_path(name)
        path.write_bytes(b"prepared weights")
        path.with_suffix(".sha256").write_text(hashlib.sha256(b"prepared weights").hexdigest())
        return path

    monkeypatch.setattr("based_thermals.model_exports.prepare_model", prepare)
    manager = model_downloads.ModelDownloads()
    try:
        assert not manager.request(model)
        assert started.wait(2)
        assert not manager.request(model, retry=True)
        assert "Preparing" in manager.status() and "verifying" in manager.status()
        assert calls == [model]
        with pytest.raises(ValueError, match="finish"):
            manager.clear_cache()
    finally:
        release.set()
    wait_finished(manager, model)
    assert manager.request(model)
    assert manager.request(model)
    assert not manager.status() and calls == [model]


def test_failed_preparation_waits_for_explicit_retry(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """
    Contain converter failures without restarting dependency setup on every frame.
    """
    monkeypatch.setenv("BASED_THERMALS_MODEL_DIR", str(tmp_path))
    calls: list[str] = []

    def prepare(name: str, progress: Callable[[str], None]) -> Path:
        """
        Fail once, then publish a prepared fixture on explicit retry.
        """
        calls.append(name)
        progress("Checking conversion")
        if len(calls) == 1:
            raise ValueError("conversion unavailable")
        path = onnx_models.model_path(name)
        path.write_bytes(b"prepared")
        path.with_suffix(".sha256").write_text(hashlib.sha256(b"prepared").hexdigest())
        return path

    monkeypatch.setattr("based_thermals.model_exports.prepare_model", prepare)
    manager = model_downloads.ModelDownloads()
    assert not manager.request("dncnn-25")
    wait_finished(manager, "dncnn-25")
    for _ in range(3):
        with pytest.raises(ValueError, match="preparation failed.*conversion unavailable"):
            manager.request("dncnn-25")
    assert calls == ["dncnn-25"]
    assert not manager.request("dncnn-25", retry=True)
    wait_finished(manager, "dncnn-25")
    assert manager.request("dncnn-25") and calls == ["dncnn-25", "dncnn-25"]


@pytest.mark.parametrize("model,factor", [
    ("style-candy", 1), ("mewzoom", 4), ("dncnn-25", 1), ("ffdnet-gray", 1),
])
def test_pending_download_keeps_feed_live_without_starting_native_helper(
    monkeypatch: pytest.MonkeyPatch, model: str, factor: int,
) -> None:
    """
    Preserve live images and geometry while downloads or denoiser conversion are pending.
    """
    requests = []

    def request(requested_model: str, retry: bool = False) -> bool:
        """
        Keep the model pending without launching inference or conversion.
        """
        requests.append((requested_model, retry))
        return False

    # noinspection PyUnresolvedReferences
    monkeypatch.setattr("based_thermals.onnx_upsampling.MODEL_DOWNLOADS.request", request)
    engine = ONNXUpsampler()
    image = np.full((12, 16, 3), 70, np.uint8)
    assert engine.apply(image, model, amount=0) is image
    assert requests == []
    for _ in range(2):
        result = engine.apply(image, model)
        assert result.shape == (12 * factor, 16 * factor, 3)
        assert np.all(result == 70)
        assert engine.process is None and not engine.error
    assert requests == [(model, True), (model, False)]


def test_missing_checksum_sidecar_triggers_verified_reinstallation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """
    Reverify cached weights when their checksum evidence is missing.
    """
    monkeypatch.setenv("BASED_THERMALS_MODEL_DIR", str(tmp_path))
    onnx_models.model_path("style-udnie").write_bytes(b"unverified")
    # noinspection PyUnresolvedReferences
    monkeypatch.setattr(model_downloads, "download_model", lambda *args, **kwargs: None)
    manager = model_downloads.ModelDownloads()
    assert not manager.request("style-udnie")
    wait_finished(manager, "style-udnie")
