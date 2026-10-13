"""Deduplicated background model installation, shared by pipeline branches."""

import logging
from threading import Lock, Thread
from typing import Any

from .onnx_models import MODELS, download_model, install_command, model_path

LOG = logging.getLogger(__name__)


class ModelDownloads:
    def __init__(self) -> None:
        """
        Share background jobs and progress across built-in and custom consumers.
        """
        self.lock = Lock()
        self.jobs: dict[str, dict[str, str]] = {}

    def request(self, model: str, retry: bool = False, *, spec: dict[str, Any] | None = None) -> bool:
        """
        Return readiness immediately; never wait for network or export Torch models.
        """
        definition = MODELS[model] if spec is None else spec
        with self.lock:
            job = self.jobs.get(model)
            if job is not None and job["state"] == "pending":
                return False
            if job is not None and job["state"] == "error" and not retry:
                raise ValueError(job["message"])
            path = model_path(model, **({"spec": spec} if spec is not None else {}))
            if path.is_file() and (
                definition["sha256"] is not None or path.with_suffix(".sha256").is_file()
            ):
                self.jobs.pop(model, None)
                return True  # The inference worker still validates the full checksum.
            if definition.get("local_export"):
                raise ValueError("Model needs a one-time local export; run " + install_command(model))
            title = definition.get("name", model)
            self.jobs[model] = {"state": "pending", "message": f"Downloading {title}…"}
            Thread(target=self._install, args=(model, spec), name=f"download-{model}", daemon=True).start()
            return False

    def _install(self, model: str, spec: dict[str, Any] | None = None) -> None:
        """
        Install in a background thread and contain download errors for UI reporting.
        """
        definition = MODELS[model] if spec is None else spec
        title = definition.get("name", model)

        def progress(received: int, total: int) -> None:
            """
            Publish byte progress through the viewer's existing status channel.
            """
            detail = f"{received / 1_000_000:.1f} MB"
            if total > 0:
                detail += f" / {total / 1_000_000:.1f} MB"
            with self.lock:
                self.jobs[model]["message"] = f"Downloading {title}: {detail}"

        try:
            if definition.get("noncommercial"):
                LOG.warning("%s weights are for non-commercial use only; see upstream terms in README", model)
            LOG.info("Installing model on demand: %s", model)
            download_model(model, progress=progress, **({"spec": spec} if spec is not None else {}))
            with self.lock:
                self.jobs[model] = {"state": "ready", "message": ""}
            LOG.info("Model ready: %s", model)
        except Exception as exc:  # noqa: BLE001 - contain background installation failures
            message = f"Model download failed ({title}): {exc}. Bypass/re-enable the node to retry."
            with self.lock:
                self.jobs[model] = {"state": "error", "message": message}
            LOG.warning("%s", message)

    def status(self) -> str:
        """
        Combine active/error job messages for the existing desktop download status.
        """
        with self.lock:
            return " · ".join(job["message"] for job in self.jobs.values() if job["message"])

    def clear_cache(self) -> tuple[int, int]:
        """
        Delete downloaded files under the job lock, refusing active installations.
        """
        from .model_cache import clear_downloaded_models

        with self.lock:
            if any(job["state"] == "pending" for job in self.jobs.values()):
                raise ValueError("Wait for model downloads to finish before clearing the cache")
            result = clear_downloaded_models()
            self.jobs.clear()
            return result


MODEL_DOWNLOADS = ModelDownloads()
