"""Deduplicated background model installation, shared by pipeline branches."""

import logging
from threading import Lock, Thread

from .onnx_models import MODELS, download_model, install_command, model_path

LOG = logging.getLogger(__name__)


class ModelDownloads:
    def __init__(self):
        self.lock = Lock()
        self.jobs = {}

    def request(self, model, retry=False):
        """Return readiness immediately; never wait for network or export Torch models."""
        with self.lock:
            job = self.jobs.get(model)
            if job and job["state"] == "pending":
                return False
            if job and job["state"] == "error" and not retry:
                raise ValueError(job["message"])
            path = model_path(model)
            if path.is_file() and (
                MODELS[model]["sha256"] is not None or path.with_suffix(".sha256").is_file()
            ):
                self.jobs.pop(model, None)
                return True  # The inference worker still validates the full checksum.
            if MODELS[model].get("local_export"):
                raise ValueError("Model needs a one-time local export; run " + install_command(model))
            self.jobs[model] = {"state": "pending", "message": f"Downloading {model}…"}
            Thread(target=self._install, args=(model,), name=f"download-{model}", daemon=True).start()
            return False

    def _install(self, model):
        def progress(received, total):
            detail = f"{received / 1_000_000:.1f} MB"
            if total > 0:
                detail += f" / {total / 1_000_000:.1f} MB"
            with self.lock:
                self.jobs[model]["message"] = f"Downloading {model}: {detail}"

        try:
            if MODELS[model].get("noncommercial"):
                LOG.warning("%s weights are for non-commercial use only; see upstream terms in README", model)
            LOG.info("Installing model on demand: %s", model)
            download_model(model, progress=progress)
            with self.lock:
                self.jobs[model] = {"state": "ready", "message": ""}
            LOG.info("Model ready: %s", model)
        except Exception as exc:  # noqa: BLE001 - contain background installation failures
            message = f"Model download failed ({model}): {exc}. Bypass/re-enable the node to retry."
            with self.lock:
                self.jobs[model] = {"state": "error", "message": message}
            LOG.warning("%s", message)

    def status(self):
        with self.lock:
            return " · ".join(job["message"] for job in self.jobs.values() if job["message"])


MODEL_DOWNLOADS = ModelDownloads()
