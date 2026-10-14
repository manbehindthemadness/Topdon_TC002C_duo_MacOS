"""
Verify custom package access to shared automatic denoiser preparation without live downloads.
"""

import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from threading import Event
from time import monotonic, sleep

import numpy as np
import pytest
from support.pipeline import raw_pipeline

from based_thermals import model_downloads, model_exports, onnx_models
from based_thermals.custom_nodes import models
from based_thermals.custom_nodes.bundle import load_folder
from based_thermals.custom_nodes.runtime import CustomProcessor
from based_thermals.pipeline import node, validate_pipeline


def wait_finished(manager: model_downloads.ModelDownloads, model: str) -> None:
    """
    Wait for a bounded fake background preparation job to finish.
    """
    deadline = monotonic() + 2
    while monotonic() < deadline:
        with manager.lock:
            if manager.jobs[model]["state"] != "pending":
                return
        sleep(0.001)
    pytest.fail("custom preparation did not finish")


def install_fixture(path: Path) -> None:
    """
    Publish checksum-valid temporary weights without executing inference.
    """
    data = b"prepared custom model"
    path.write_bytes(data)
    path.with_suffix(".sha256").write_text(hashlib.sha256(data).hexdigest())


@pytest.mark.parametrize("model", model_exports.EXPORT_MODELS)
def test_custom_worker_shares_conversion_with_builtin_nodes_and_reuses_cache(
    model: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """
    Validate/export inert packages, then prepare once for independent workers and built-in consumers.
    """
    cache, folder = tmp_path / "cache", tmp_path / "package"
    folder.mkdir()
    monkeypatch.setenv("BASED_THERMALS_MODEL_DIR", str(cache))
    manager = model_downloads.ModelDownloads()
    # noinspection PyUnresolvedReferences
    monkeypatch.setattr(models, "MODEL_DOWNLOADS", manager)
    monkeypatch.setattr("based_thermals.runtime_setup.install.find_uv", lambda: "/fake/uv")
    started, release = Event(), Event()
    calls: list[str] = []

    def convert(
        command: list[str],
        environment: dict[str, str],
        directory: Path,
        **_kwargs: object,
    ) -> None:
        """
        Hold conversion while Custom and built-in consumers continue polling.
        """
        calls.append(command[-1])
        assert environment["BASED_THERMALS_MODEL_DIR"] == str(directory)
        started.set()
        assert release.wait(2)
        install_fixture(onnx_models.model_path(model))

    monkeypatch.setattr(model_exports, "run_export", convert)
    (folder / "__init__.py").write_text(
        "from based_thermals.custom_nodes.models import ModelResolver\n"
        f"MODEL_SOURCE = {model!r}\n"
        "models = ModelResolver()\n"
        "def process(image, config):\n"
        "    path = models.resolve(config['model'])\n"
        "    return image if path is None else image + 10\n"
    )
    item = node("software", "custom", **load_folder(folder))
    document = validate_pipeline(json.loads(json.dumps(raw_pipeline(item))))
    item = document["software"][2]
    assert json.loads(item["params"]["config"])["model"] == model
    assert not calls and not manager.jobs and not cache.exists()
    first, second = CustomProcessor(), CustomProcessor()
    image = np.full((3, 4, 3), 20, dtype=np.float32)
    try:
        assert np.array_equal(first.apply(image, item), image)
        assert started.wait(2)
        assert np.array_equal(second.apply(image, item), image)
        assert not manager.request(model, retry=True)
        assert calls == [model] and "Preparing" in manager.status()
        release.set()
        wait_finished(manager, model)
        assert np.array_equal(first.apply(image, item), image + 10)
        assert np.array_equal(second.apply(image, item), image + 10)
        assert manager.request(model) and calls == [model]
        assert not manager.status()

        offline = model_downloads.ModelDownloads()
        # noinspection PyUnresolvedReferences
        monkeypatch.setattr(models, "MODEL_DOWNLOADS", offline)
        resolver = models.ModelResolver()
        assert resolver.resolve(model) is None
        wait_finished(offline, model)
        assert resolver.resolve(model) == onnx_models.model_path(model)
        assert calls == [model]
    finally:
        release.set()
        wait_finished(manager, model)
        first.close()
        second.close()


@pytest.mark.parametrize("model", model_exports.EXPORT_MODELS)
def test_custom_conversion_failure_retries_once_with_new_resolver(
    model: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """
    Preserve conversion errors during polling and permit package recreation to retry once.
    """
    monkeypatch.setenv("BASED_THERMALS_MODEL_DIR", str(tmp_path))
    manager = model_downloads.ModelDownloads()
    # noinspection PyUnresolvedReferences
    monkeypatch.setattr(models, "MODEL_DOWNLOADS", manager)
    calls: list[str] = []

    def prepare(name: str, progress: Callable[[str], None]) -> Path:
        """
        Fail the initial conversion, then publish verified weights on retry.
        """
        calls.append(name)
        progress("Converting")
        if len(calls) == 1:
            raise ValueError("converter offline")
        path = onnx_models.model_path(name)
        install_fixture(path)
        return path

    monkeypatch.setattr(model_exports, "prepare_model", prepare)
    resolver = models.ModelResolver()
    assert resolver.resolve(model) is None
    wait_finished(manager, model)
    for _ in range(3):
        with pytest.raises(ValueError, match="preparation failed.*converter offline"):
            resolver.resolve(model)
    assert calls == [model]
    recreated = models.ModelResolver()
    assert recreated.resolve(model) is None
    wait_finished(manager, model)
    assert recreated.resolve(model) == onnx_models.model_path(model)
    assert resolver.resolve(model) == onnx_models.model_path(model)
    assert calls == [model, model]
