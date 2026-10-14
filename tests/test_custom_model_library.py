"""
Local model discovery and path registration without inference or network access.
"""

import json
from pathlib import Path

import pytest

from based_thermals.custom_nodes.model_library import (
    add_model,
    existing_models,
    library_path,
    named_model_choices,
)


def test_cached_registered_and_current_models_are_discovered(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Share browsed paths across nodes without copying binaries or dropping missing selections.
    """
    cache = tmp_path / "cache"
    cache.mkdir()
    monkeypatch.setenv("BASED_THERMALS_MODEL_DIR", str(cache))
    cached = cache / "cached.onnx"
    cached.write_bytes(b"cached")
    external = tmp_path / "external.onnx"
    external.write_bytes(b"weights")
    value = add_model(external)
    assert add_model(external) == value
    assert json.loads(library_path().read_text()) == [value]
    paths = existing_models()
    assert str(cached) in paths and value in paths
    assert not (cache / "external.onnx").exists()
    external.unlink()
    assert value not in existing_models()
    assert value in existing_models(value)


def test_invalid_browse_selection_and_corrupt_registry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Ignore broken discovery metadata and refuse nonexistent or non-ONNX files.
    """
    monkeypatch.setenv("BASED_THERMALS_MODEL_DIR", str(tmp_path))
    library_path().write_text('{"wrong": "shape"}')
    assert isinstance(existing_models(), list)
    for path in (tmp_path / "missing.onnx", tmp_path / "text.txt"):
        if path.suffix == ".txt":
            path.write_text("text")
        with pytest.raises(ValueError, match="existing ONNX"):
            add_model(path)
    assert json.loads(library_path().read_text()) == {"wrong": "shape"}


def test_named_choices_filter_cache_and_retain_explicit_overrides(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Name matching cached/original files without opening weights, and retain missing saved paths.
    """
    monkeypatch.setenv("BASED_THERMALS_MODEL_DIR", str(tmp_path))
    detector = tmp_path / "custom-new-source-155c3c6f37dc.onnx"
    detector.touch()
    original = tmp_path / "head-detector.onnx"
    original.touch()
    unrelated = tmp_path / "acnet-legacy-hdn0.onnx"
    unrelated.touch()
    models = [{"name": "HGNet-N head detector", "sha256": "155c3c6f37dc" + "0" * 52,
               "filenames": [original.name]}]
    choices = named_model_choices(models=models)
    assert set(choices) == {("HGNet-N head detector", str(detector)),
                            ("HGNet-N head detector", str(original))}
    assert str(unrelated) not in [path for _label, path in choices]
    missing = tmp_path / "local-override.onnx"
    choices = named_model_choices(str(missing), models)
    assert ("Local override: local-override.onnx — missing", str(missing)) in choices
    assert str(unrelated) in [path for _label, path in named_model_choices()]
