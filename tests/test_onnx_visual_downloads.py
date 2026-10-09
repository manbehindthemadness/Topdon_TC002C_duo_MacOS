import hashlib
import io
import ssl
import sys
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pytest

from topdon_duo import onnx_models


def test_denoiser_download_needs_export_and_no_network(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("TOPDON_MODEL_DIR", str(tmp_path))
    opener = Mock(side_effect=AssertionError("no network"))
    monkeypatch.setattr(onnx_models.urllib.request, "urlopen", opener)
    with pytest.raises(ValueError, match="export_visual_denoisers.py dncnn-25"):
        onnx_models.download_model("dncnn-25")
    with pytest.raises(ValueError, match="export_visual_denoisers.py ffdnet-gray"):
        onnx_models.verified_model("ffdnet-gray")
    opener.assert_not_called()


def test_verified_download_is_atomic_reused_and_rejects_wrong_content(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("TOPDON_MODEL_DIR", str(tmp_path))
    data = b"test model fixture"
    spec = dict(onnx_models.MODELS["espcn"], sha256=hashlib.sha256(data).hexdigest())
    monkeypatch.setitem(onnx_models.MODELS, "espcn", spec)
    opener = Mock(side_effect=lambda *_args, **_kwargs: io.BytesIO(data))
    monkeypatch.setattr(onnx_models.urllib.request, "urlopen", opener)
    path = onnx_models.download_model("espcn")
    assert onnx_models.verified_model("espcn") == data
    assert onnx_models.download_model("espcn") == path
    assert opener.call_count == 1
    context = opener.call_args.kwargs["context"]
    assert context.verify_mode == ssl.CERT_REQUIRED
    assert context.check_hostname
    monkeypatch.setitem(onnx_models.MODELS, "mewzoom", dict(spec, sha256="0" * 64))
    with pytest.raises(ValueError, match="checksum"):
        onnx_models.download_model("mewzoom")
    assert not onnx_models.model_path("mewzoom").exists()
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize(
    "platform,custom_ca", [("darwin", False), ("darwin", True), ("linux", False)]
)
def test_download_trust_bundle_is_mac_only_and_respects_custom_ca(
    monkeypatch: pytest.MonkeyPatch, platform: Any, custom_ca: Any
) -> None:
    monkeypatch.setattr(sys, "platform", platform)
    monkeypatch.delenv("SSL_CERT_FILE", raising=False)
    monkeypatch.delenv("SSL_CERT_DIR", raising=False)
    if custom_ca:
        monkeypatch.setenv("SSL_CERT_FILE", "/custom/trusted-ca.pem")
    context = Mock()
    monkeypatch.setattr(onnx_models.ssl, "create_default_context", lambda: context)
    monkeypatch.setattr(onnx_models.Path, "is_file", lambda _path: True)
    assert onnx_models.download_ssl_context() is context
    if platform == "darwin" and not custom_ca:
        context.load_verify_locations.assert_called_once_with(cafile="/etc/ssl/cert.pem")
    else:
        context.load_verify_locations.assert_not_called()


def test_v1_installer_verifies_publisher_pointer_and_retains_checksum_offline(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("TOPDON_MODEL_DIR", str(tmp_path))
    data = b"v1 test model"
    checksum = hashlib.sha256(data).hexdigest()
    pointer = f"version https://git-lfs.github.com/spec/v1\noid sha256:{checksum}\nsize {len(data)}\n".encode()
    opener = Mock(side_effect=[io.BytesIO(pointer), io.BytesIO(data)])
    monkeypatch.setattr(onnx_models.urllib.request, "urlopen", opener)
    path = onnx_models.download_model("mewzoom-v1-2x")
    assert "/raw/main/model.onnx" in opener.call_args_list[0].args[0]
    assert "/resolve/main/model.onnx" in opener.call_args_list[1].args[0]
    assert path.with_suffix(".sha256").read_text().strip() == checksum
    assert onnx_models.verified_model("mewzoom-v1-2x") == data
    assert onnx_models.download_model("mewzoom-v1-2x") == path
    assert opener.call_count == 2  # Cached weights need no remote metadata request.


@pytest.mark.parametrize(
    "pointer",
    [
        b"not an LFS pointer",
        b"version https://git-lfs.github.com/spec/v1\noid sha256:"
        + b"a" * 64
        + b"\nsize 999999999\n",
    ],
)
def test_v1_installer_rejects_invalid_or_oversized_publisher_metadata(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, pointer: Any
) -> None:
    monkeypatch.setenv("TOPDON_MODEL_DIR", str(tmp_path))
    opener = Mock(return_value=io.BytesIO(pointer))
    monkeypatch.setattr(onnx_models.urllib.request, "urlopen", opener)
    with pytest.raises(ValueError):
        onnx_models.download_model("mewzoom-v1-4x")
    assert opener.call_count == 1
    assert list(tmp_path.iterdir()) == []


def test_missing_model_is_actionable(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("TOPDON_MODEL_DIR", str(tmp_path))
    with pytest.raises(ValueError, match="uv run topdon-duo-models mewzoom"):
        onnx_models.verified_model("mewzoom")
