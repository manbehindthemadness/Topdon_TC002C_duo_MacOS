from threading import Event
from time import monotonic, sleep

import numpy as np
import pytest

from topdon_duo import model_downloads, onnx_models
from topdon_duo.onnx_upsampling import ONNXUpsampler


def wait_finished(manager, model):
    deadline = monotonic() + 2
    while monotonic() < deadline:
        with manager.lock:
            if manager.jobs[model]["state"] != "pending":
                return
        sleep(0.005)
    pytest.fail("background installation did not finish")


def test_background_download_deduplicates_branches_and_reuses_offline_cache(monkeypatch, tmp_path):
    monkeypatch.setenv("TOPDON_MODEL_DIR", str(tmp_path))
    started, release = Event(), Event()
    calls = []

    def install(model, progress):
        calls.append(model)
        progress(1_000_000, 2_000_000)
        started.set()
        assert release.wait(2)
        onnx_models.model_path(model).write_bytes(b"test weights")

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


def test_failed_download_does_not_retry_every_frame_but_can_retry(monkeypatch, tmp_path):
    monkeypatch.setenv("TOPDON_MODEL_DIR", str(tmp_path))
    calls = []

    def install(model, progress):
        calls.append(model)
        raise OSError("offline")

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


def test_local_export_is_never_automatically_started(monkeypatch, tmp_path):
    monkeypatch.setenv("TOPDON_MODEL_DIR", str(tmp_path))
    manager = model_downloads.ModelDownloads()
    with pytest.raises(ValueError, match="export_visual_denoisers"):
        manager.request("ffdnet-gray")
    assert not manager.jobs


@pytest.mark.parametrize("model,factor", [("style-candy", 1), ("mewzoom", 4)])
def test_pending_download_keeps_feed_live_without_starting_native_helper(monkeypatch, model, factor):
    requests = []

    def request(model, retry=False):
        requests.append((model, retry))
        return False

    monkeypatch.setattr("topdon_duo.onnx_upsampling.MODEL_DOWNLOADS.request", request)
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


def test_missing_checksum_sidecar_triggers_verified_reinstallation(monkeypatch, tmp_path):
    monkeypatch.setenv("TOPDON_MODEL_DIR", str(tmp_path))
    onnx_models.model_path("style-udnie").write_bytes(b"unverified")
    monkeypatch.setattr(model_downloads, "download_model", lambda *args, **kwargs: None)
    manager = model_downloads.ModelDownloads()
    assert not manager.request("style-udnie")
    wait_finished(manager, "style-udnie")
