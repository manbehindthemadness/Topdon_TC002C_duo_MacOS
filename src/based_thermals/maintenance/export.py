"""
Portable support ZIP snapshots without importing nodes or probing hardware/network services.
"""

from __future__ import annotations

import json
import os
import platform
import sys
import tempfile
from collections.abc import Callable
from datetime import UTC, datetime
from importlib.metadata import distributions
from pathlib import Path
from threading import Thread
from typing import Any
from zipfile import ZIP_DEFLATED, ZipFile

from .. import __version__
from ..desktop_app.captures import restore_user_ownership
from ..dialog_preferences import load_dialog_directory
from .paths import MaintenancePaths

SETTINGS_FILES = {
    "settings.json",
    "main-window.json",
    "pipeline-presets.json",
    "pipeline-selection.json",
    "desktop.ini",
    "maintenance.json",
}
LOG_SUFFIXES = {".log", ".jsonl", ".bin", ".csv"}


def support_metadata() -> dict[str, Any]:
    """
    Describe the running Python/platform/packages without environment variables or active probes.
    """
    packages = sorted(
        {f"{item.metadata.get('Name', 'unknown')}=={item.version}" for item in distributions()}
    )
    return {
        "created_utc": datetime.now(UTC).isoformat(),
        "application_version": __version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": sys.version,
        "executable": sys.executable,
        "packages": packages,
    }


def export_files(
    paths: MaintenancePaths,
    all_data: bool,
    diagnostics: Path | None,
) -> dict[str, Path]:
    """
    Collect only app settings and log files, excluding source, images, models and directory links.
    """
    files: dict[str, Path] = {}
    for index, root in enumerate(paths.settings):
        if root.is_symlink() or not root.is_dir():
            continue
        for name in SETTINGS_FILES:
            files[f"settings/{index}/{name}"] = root / name
        for folder in ("camera-profiles", "dialog-directories"):
            directory = root / folder
            if directory.is_dir() and not directory.is_symlink():
                for path in sorted(directory.glob("*.json")):
                    files[f"settings/{index}/{folder}/{path.name}"] = path
    if not paths.models.is_symlink():
        files["settings/custom-models.json"] = paths.models / "custom-models.json"
    if not all_data:
        return files
    if not paths.runtimes.is_symlink():
        files["debug/runtime-state.json"] = paths.runtimes / "state.json"
    baselines = paths.state / "camera-baselines"
    if baselines.is_dir() and not baselines.is_symlink():
        for path in sorted(baselines.glob("*.json")):
            files[f"debug/camera-baselines/{path.name}"] = path
    for index, path in enumerate(paths.artifacts):
        if path.name.startswith(":memory:.ses"):
            files[f"debug/session-artifacts/{index}-{path.name.replace(':', '_')}"] = path
    roots = [paths.state / "debug-logs"]
    roots.extend(path for path in paths.outputs if path.name == "diagnostics")
    temperature_logs = load_dialog_directory("temperatures")
    if temperature_logs is not None:
        roots.append(temperature_logs)
    for index, root in enumerate(dict.fromkeys(roots)):
        if root.is_symlink() or not root.is_dir():
            continue
        for path in sorted(root.iterdir()):
            if path.suffix in LOG_SUFFIXES or path.name.startswith("application.log."):
                files[f"logs/{index}/{path.name}"] = path
    # Also include default-named temperature logs in ordinary output directories.
    for index, root in enumerate(paths.outputs):
        if root.is_symlink() or not root.is_dir():
            continue
        for path in sorted(root.glob("*.csv")):
            if path.name.startswith(("based-thermals-", "topdon-duo-")):
                files[f"temperature-logs/{index}/{path.name}"] = path
    if diagnostics is not None and not diagnostics.parent.is_symlink():
        files[f"diagnostics/{diagnostics.name}"] = diagnostics
        for path in sorted(diagnostics.parent.glob(diagnostics.stem + ".*")):
            if path.suffix in LOG_SUFFIXES:
                files[f"diagnostics/{path.name}"] = path
    return files


def create_archive(
    destination: Path,
    paths: MaintenancePaths,
    snapshot: dict[str, Any],
    *,
    all_data: bool,
    diagnostics: Path | None = None,
) -> tuple[int, list[str]]:
    """
    Atomically publish a ZIP with saved originals, current state and a manifest of read failures.
    """
    if destination.suffix.lower() != ".zip" or not destination.is_absolute():
        raise ValueError("Choose an absolute .zip archive filename")
    if destination.is_symlink():
        raise ValueError("Cannot replace a linked archive")
    candidates = export_files(paths, all_data, diagnostics)
    temporary: Path | None = None
    problems: list[str] = []
    included: dict[str, str] = {}
    try:
        with tempfile.NamedTemporaryFile(
            dir=destination.parent,
            prefix=".based-support-",
            suffix=".zip",
            delete=False,
        ) as stream:
            created = Path(stream.name)
            temporary = created
        with ZipFile(created, "w", compression=ZIP_DEFLATED) as archive:
            archive.writestr("current-settings.json", json.dumps(snapshot["settings"], indent=2))
            if all_data:
                archive.writestr("system.json", json.dumps(support_metadata(), indent=2))
                archive.writestr("session.json", json.dumps(snapshot["session"], indent=2))
            seen: set[Path] = set()
            for name, path in candidates.items():
                if path.is_symlink():
                    problems.append(f"Skipped linked file: {path}")
                    continue
                if not path.is_file() or path.resolve() in seen or path.resolve() == destination:
                    continue
                try:
                    # Fix the read length so a growing active log cannot keep export running forever.
                    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
                    with os.fdopen(descriptor, "rb") as source, archive.open(name, "w") as output:
                        remaining = os.fstat(source.fileno()).st_size
                        while remaining:
                            chunk = source.read(min(remaining, 1024 * 1024))
                            if not chunk:
                                break
                            output.write(chunk)
                            remaining -= len(chunk)
                    included[name] = str(path)
                    seen.add(path.resolve())
                except OSError as exc:
                    problems.append(f"Could not fully read {path}: {exc}")
            archive.writestr(
                "manifest.json",
                json.dumps(
                    {
                        "format": 1,
                        "kind": "all" if all_data else "settings",
                        "files": included,
                        "warnings": problems,
                        "notes": "Saved originals and current state are separate. Active logs are snapshots. "
                        "No captures, model weights, runtime binaries or external node/backend "
                        "source folders are included. Review settings/logs before sharing.",
                    },
                    indent=2,
                ),
            )
        created.replace(destination)
        restore_user_ownership([destination])
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return len(included), problems


class SupportExport:
    """
    Run one support export off the frame loop and publish its result only after completion.
    """

    def __init__(self) -> None:
        """
        Initialize passive state for a single viewer session.
        """
        self.thread: Thread | None = None
        self.result: str | None = None

    @property
    def busy(self) -> bool:
        """
        Report an active worker so cleanup cannot race with its reads.
        """
        return self.thread is not None and self.thread.is_alive()

    def start(
        self,
        destination: Path,
        paths: MaintenancePaths,
        snapshot: dict[str, Any],
        *,
        all_data: bool,
        diagnostics: Path | None,
        flush_logs: Callable[[], None] | None = None,
    ) -> None:
        """
        Start an already-copied configuration snapshot without blocking capture.
        """
        if self.busy:
            raise ValueError("A support export is already running")
        self.result = None

        def work() -> None:
            """
            Finish the archive and retain a readable success/partial/failure message.
            """
            try:
                if flush_logs is not None:
                    flush_logs()
                count, warnings = create_archive(
                    destination,
                    paths,
                    snapshot,
                    all_data=all_data,
                    diagnostics=diagnostics,
                )
                message = f"Exported {destination.name}: {count} saved files"
                if warnings:
                    message += f" ({len(warnings)} warnings; see manifest.json)"
                self.result = message
            except (OSError, ValueError) as exc:
                self.result = f"Support export failed: {exc}"

        thread = Thread(target=work, name="support-export", daemon=True)
        self.thread = thread
        thread.start()

    def poll(self) -> str | None:
        """
        Consume a completed result once; the worker never updates Qt or session state.
        """
        if self.busy:
            return None
        result, self.result = self.result, None
        return result

    def close(self) -> None:
        """
        Let the bounded file snapshot finish before the viewer exits or removes data.
        """
        if self.thread is not None:
            self.thread.join()
