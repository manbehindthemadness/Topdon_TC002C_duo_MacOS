"""
Checksum-pinned experiment downloads remain independent of live network access.
"""

import hashlib
import importlib.util
import io
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest


@pytest.fixture
def preparation() -> ModuleType:
    """
    Import the setup script without installing or importing its isolated ONNX dependencies.
    """
    path = Path(__file__).parents[1] / "tools/prepare_head_pose.py"
    spec = importlib.util.spec_from_file_location("head_pose_preparation_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_verified_download_is_reused_without_network(
    preparation: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Offline preparation reuses exact bytes and rejects altered files.
    """
    path = tmp_path / "model.onnx"
    path.write_bytes(b"verified bytes")
    checksum = hashlib.sha256(b"verified bytes").hexdigest()

    def forbidden(*_args: Any, **_kwargs: Any) -> None:
        """
        Fail if a verified local asset contacts the network.
        """
        raise AssertionError("Unexpected network request")

    monkeypatch.setitem(preparation.__dict__, "urlopen", forbidden)
    preparation.download(path, "model.onnx", checksum, True)
    path.write_bytes(b"modified bytes")
    with pytest.raises(ValueError, match="Missing or changed"):
        preparation.download(path, "model.onnx", checksum, True)


@pytest.mark.parametrize("matches", [True, False])
def test_download_checks_bytes_before_replacing_previous_asset(
    preparation: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, matches: bool,
) -> None:
    """
    Reject changed upstream assets and preserve an older local file on failure.
    """
    path = tmp_path / "model.onnx"
    path.write_bytes(b"previous bytes")
    payload = b"downloaded bytes"
    checksum = hashlib.sha256(payload if matches else b"different expected bytes").hexdigest()
    monkeypatch.setitem(preparation.__dict__, "importlib", SimpleNamespace(
        import_module=lambda _name: SimpleNamespace(where=lambda: "fake-ca.pem"),
    ))
    monkeypatch.setitem(preparation.__dict__, "ssl", SimpleNamespace(
        create_default_context=lambda **_kwargs: object(),
    ))

    def response(url: str, *, timeout: int, context: object) -> io.BytesIO:
        """
        Verify HTTPS is retained and provide a bounded in-memory network response.
        """
        assert url.startswith("https://") and timeout == 60 and context is not None
        return io.BytesIO(payload)

    monkeypatch.setitem(preparation.__dict__, "urlopen", response)
    if matches:
        preparation.download(path, "model.onnx", checksum, False)
        assert path.read_bytes() == payload
    else:
        with pytest.raises(ValueError, match="checksum changed"):
            preparation.download(path, "model.onnx", checksum, False)
        assert path.read_bytes() == b"previous bytes"
    assert not path.with_suffix(".onnx.partial").exists()
