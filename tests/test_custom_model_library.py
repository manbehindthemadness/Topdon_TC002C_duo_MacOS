"""
Local model discovery and path registration without inference or network access.
"""

import json
from pathlib import Path

import pytest

from topdon_duo.custom_nodes.model_library import add_model, existing_models, library_path


def test_cached_registered_and_current_models_are_discovered(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Share browsed paths across nodes without copying binaries or dropping missing selections.
    """
    cache = tmp_path / "cache"
    cache.mkdir()
    monkeypatch.setenv("TOPDON_MODEL_DIR", str(cache))
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
    monkeypatch.setenv("TOPDON_MODEL_DIR", str(tmp_path))
    library_path().write_text('{"wrong": "shape"}')
    assert isinstance(existing_models(), list)
    for path in (tmp_path / "missing.onnx", tmp_path / "text.txt"):
        if path.suffix == ".txt":
            path.write_text("text")
        with pytest.raises(ValueError, match="existing ONNX"):
            add_model(path)
    assert json.loads(library_path().read_text()) == {"wrong": "shape"}
