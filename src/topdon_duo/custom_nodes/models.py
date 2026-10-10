"""
Nonblocking access to the viewer's shared downloader from custom Python packages.
"""

import hashlib
import json
import re
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urlsplit

from ..model_downloads import MODEL_DOWNLOADS
from ..onnx_models import MODELS, model_path, verified_model


def model_source(source: str | dict[str, Any]) -> tuple[str, dict[str, Any] | None]:
    """
    Resolve catalog names or validate an HTTPS, checksum-pinned custom source.
    """
    if isinstance(source, str):
        if source not in MODELS:
            raise ValueError(f"Unknown model: {source}")
        return source, None
    allowed = {"name", "url", "sha256", "archive_member", "archive_sha256", "max_bytes"}
    if not isinstance(source, dict) or set(source) - allowed or not {"url", "sha256"} <= set(source):
        raise ValueError("Custom model source requires url/sha256 and supported optional fields")
    spec = dict(source)
    url, checksum = spec["url"], spec["sha256"]
    if not isinstance(url, str) or urlsplit(url).scheme != "https" or not urlsplit(url).hostname:
        raise ValueError("Custom model URL must use HTTPS")
    if not isinstance(checksum, str) or not re.fullmatch("[0-9a-f]{64}", checksum):
        raise ValueError("Custom model needs a pinned lowercase SHA-256 checksum")
    name = spec.setdefault("name", "Custom model")
    if not isinstance(name, str) or not name.strip() or len(name) > 200:
        raise ValueError("Invalid custom model name")
    maximum = spec.setdefault("max_bytes", 70_000_000)
    if type(maximum) is not int or not 1 <= maximum <= 512_000_000:
        raise ValueError("Custom model max_bytes must be an integer from 1 to 512000000")
    if "archive_member" in spec or "archive_sha256" in spec:
        member = spec.get("archive_member")
        archive_checksum = spec.get("archive_sha256")
        if not isinstance(member, str) or not member or "\\" in member:
            raise ValueError("Invalid custom model archive member")
        path = PurePosixPath(member)
        if path.is_absolute() or ".." in path.parts or path.as_posix() != member:
            raise ValueError("Invalid custom model archive member")
        if not isinstance(archive_checksum, str) or not re.fullmatch("[0-9a-f]{64}", archive_checksum):
            raise ValueError("Custom model archive needs its pinned SHA-256 checksum")
    identity = "custom-" + hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:24]
    return identity, spec


class ModelResolver:
    """
    Request shared background downloads and verify ready paths once per file revision.
    """

    def __init__(self) -> None:
        """
        Keep retry and verification state local to one custom package instance.
        """
        self.requested: set[str] = set()
        self.verified: dict[str, tuple[Path, int, int, int]] = {}

    def resolve(self, source: str | dict[str, Any]) -> Path | None:
        """
        Return verified cached weights, or None while the background job is pending.

        Calling again polls without blocking or retrying failed jobs every frame.
        Recreating the resolver (such as bypass/re-enable) allows one fresh retry.
        """
        identity, spec = model_source(source)
        retry = identity not in self.requested
        self.requested.add(identity)
        kwargs = {"spec": spec} if spec is not None else {}
        if not MODEL_DOWNLOADS.request(identity, retry=retry, **kwargs):
            return None
        path = model_path(identity, **kwargs)
        info = path.stat()
        revision = path, info.st_mtime_ns, info.st_ctime_ns, info.st_size
        if self.verified.get(identity) != revision:
            verified_model(identity, **kwargs)
            self.verified[identity] = revision
        return path
