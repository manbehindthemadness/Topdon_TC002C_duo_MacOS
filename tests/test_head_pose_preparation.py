"""
Checksum-pinned experiment downloads remain independent of live network access.
"""

import ast
import hashlib
import importlib.util
import io
import json
import sys
import tarfile
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


@pytest.mark.parametrize("variant", ["vitt-128", "vitt-128-rollgate"])
def test_prepare_selected_pose_and_record_matching_provenance(
    preparation: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, variant: str,
) -> None:
    """
    Select the requested exact archive member, check its bytes and record its source.
    """
    archive_file, archive_sha, member, _digest = preparation.POSES[variant]
    payload = b"independent fake pose bytes"
    digest = hashlib.sha256(payload).hexdigest()
    monkeypatch.setitem(preparation.POSES, variant, (archive_file, archive_sha, member, digest))
    requests = []

    def download(path: Path, filename: str, expected: str, offline: bool) -> None:
        """
        Supply a fake archive with the exact requested regular member, without network.
        """
        requests.append((filename, expected))
        assert offline
        if filename.endswith(".tar.gz"):
            with tarfile.open(path, "w:gz") as bundle:
                info = tarfile.TarInfo(member)
                info.size = len(payload)
                bundle.addfile(info, io.BytesIO(payload))
        else:
            path.write_bytes(b"detector bytes")

    monkeypatch.setitem(preparation.__dict__, "download", download)
    monkeypatch.setitem(preparation.__dict__, "prepare_detector", lambda *_args: {})
    monkeypatch.setattr(sys, "argv", ["prepare_head_pose.py", "--output", str(tmp_path),
                                    "--pose", variant, "--offline"])
    preparation.main()
    assert (tmp_path / "pose.onnx").read_bytes() == payload
    assert requests[1] == (archive_file, archive_sha)
    report = json.loads((tmp_path / "provenance.json").read_text())
    assert report["pose_member"] == member and report["pose_sha256"] == digest
    assert report["pose_archive_sha256"] == archive_sha


def test_preparation_and_portable_node_pose_pins_agree(preparation: ModuleType) -> None:
    """
    Keep standalone preparation and embedded worker sources pinned to identical releases.
    """
    path = Path(__file__).parents[1] / "examples/custom_nodes/head-pose/nodes/Head Pose/sources.py"
    tree = ast.parse(path.read_text())
    sources = next(ast.literal_eval(statement.value) for statement in tree.body
                   if isinstance(statement, ast.Assign)
                   and any(isinstance(target, ast.Name) and target.id == "POSES"
                           for target in statement.targets))
    for source in sources.values():
        expected = (Path(source["url"]).name, source["archive_sha256"],
                    source["archive_member"], source["sha256"])
        assert expected in preparation.POSES.values()
