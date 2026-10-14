"""Deduplicated background model installation, shared by pipeline branches."""

import logging
from threading import Event, Lock, Thread, current_thread
from time import monotonic
from typing import Any

from . import model_exports
from .onnx_models import MODELS, download_model, model_path

LOG = logging.getLogger(__name__)


class ModelDownloads:
    def __init__(self) -> None:
        """
        Share background jobs and progress across built-in and custom consumers.
        """
        self.lock = Lock()
        self.jobs: dict[str, dict[str, str]] = {}
        self.stopping = Event()
        self.threads: set[Thread] = set()

    def begin(self) -> None:
        """
        Reopen a fully stopped manager for an explicitly started desktop session.
        """
        with self.lock:
            if self.stopping.is_set() and not self.threads:
                self.stopping.clear()
                self.jobs.clear()

    def stop(self, timeout: float = 35.0) -> None:
        """
        Cancel background progress and wait for all installers before deleting their files.
        """
        self.stopping.set()
        with self.lock:
            threads = tuple(self.threads)
        deadline = monotonic() + timeout
        for thread in threads:
            thread.join(max(0, deadline - monotonic()))
        if any(thread.is_alive() for thread in threads):
            raise TimeoutError("Model installers are still active; maintenance was not performed")

    def request(
        self, model: str, retry: bool = False, *, spec: dict[str, Any] | None = None
    ) -> bool:
        """
        Return readiness immediately while missing weights are downloaded or prepared in background.
        """
        definition = MODELS[model] if spec is None else spec
        with self.lock:
            if self.stopping.is_set():
                return False
            job = self.jobs.get(model)
            if job is not None and job["state"] == "pending":
                return False
            if job is not None and job["state"] == "error" and not retry:
                raise ValueError(job["message"])
            path = model_path(model, **({"spec": spec} if spec is not None else {}))
            installed = path.is_file() and (
                definition["sha256"] is not None or path.with_suffix(".sha256").is_file()
            )
            local_export = bool(definition.get("local_export"))
            if installed and (not local_export or job is not None and job["state"] == "ready"):
                if not local_export:
                    self.jobs.pop(model, None)
                return True  # The inference worker still validates the full checksum.
            title = definition.get("name", model)
            action = "Preparing" if definition.get("local_export") else "Downloading"
            self.jobs[model] = {"state": "pending", "message": f"{action} {title}…"}
            thread = Thread(
                target=self._install, args=(model, spec), name=f"download-{model}", daemon=True
            )
            self.threads.add(thread)
            thread.start()
            return False

    def _install(self, model: str, spec: dict[str, Any] | None = None) -> None:
        """
        Install or prepare in a background thread and contain failures for UI reporting.
        """
        definition = MODELS[model] if spec is None else spec
        title = definition.get("name", model)

        def progress(received: int, total: int) -> None:
            """
            Publish byte progress through the viewer's existing status channel.
            """
            if self.stopping.is_set():
                raise ValueError("Model installation cancelled for maintenance")
            detail = f"{received / 1_000_000:.1f} MB"
            if total > 0:
                detail += f" / {total / 1_000_000:.1f} MB"
            with self.lock:
                self.jobs[model]["message"] = f"Downloading {title}: {detail}"

        def preparation(detail: str) -> None:
            """
            Publish isolated conversion status without blocking the processing worker.
            """
            if self.stopping.is_set():
                raise ValueError("Model preparation cancelled for maintenance")
            with self.lock:
                self.jobs[model]["message"] = f"Preparing {title}: {detail}"

        try:
            if definition.get("noncommercial"):
                LOG.warning(
                    "%s weights are for non-commercial use only; see upstream terms in README",
                    model,
                )
            LOG.info("Installing model on demand: %s", model)
            if definition.get("local_export"):
                if spec is not None:
                    raise ValueError("Custom model descriptors cannot request code conversion")
                model_exports.prepare_model(model, progress=preparation)
            else:
                download_model(
                    model, progress=progress, **({"spec": spec} if spec is not None else {})
                )
            with self.lock:
                self.jobs[model] = {"state": "ready", "message": ""}
            LOG.info("Model ready: %s", model)
        except Exception as exc:  # noqa: BLE001 - contain background installation failures
            action = "preparation" if definition.get("local_export") else "download"
            message = f"Model {action} failed ({title}): {exc}. Bypass/re-enable the node to retry."
            with self.lock:
                self.jobs[model] = {"state": "error", "message": message}
            LOG.warning("%s", message)
        finally:
            with self.lock:
                self.threads.discard(current_thread())

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
