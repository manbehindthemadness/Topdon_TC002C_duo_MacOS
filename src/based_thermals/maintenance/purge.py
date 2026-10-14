"""
Targeted cleanup that never traverses links or removes camera/custom-node source directories.
"""

from __future__ import annotations

import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

from ..model_cache import clear_downloaded_models
from .paths import MaintenancePaths

CATEGORIES = frozenset({"models", "runtimes", "settings", "sessions", "bytecode", "outputs"})
SKIP = frozenset({".git", ".venv", "venv", "standards", ".idea", ".agents", "node_modules"})


@dataclass(frozen=True)
class MaintenancePlan:
    """
    Store server-resolved targets rather than accepting paths from UI messages.
    """

    categories: frozenset[str]
    paths: MaintenancePaths

    @classmethod
    def from_command(cls, command: dict, paths: MaintenancePaths) -> MaintenancePlan:
        """
        Validate selected cleanup categories before stopping the viewer.
        """
        categories = command.get("categories")
        if (
            not isinstance(categories, list)
            or not categories
            or any(not isinstance(item, str) or item not in CATEGORIES for item in categories)
        ):
            raise ValueError("Choose valid maintenance items")
        return cls(frozenset(categories), paths)

    def execute(self) -> tuple[int, int]:
        """
        Remove selected artifacts after shutdown; report partial failures explicitly.
        """
        count, size = 0, 0
        failures = []
        targets: list[Path] = []
        if "settings" in self.categories:
            if "models" not in self.categories and not self.paths.models.is_symlink():
                targets.append(self.paths.models / "custom-models.json")
            for directory in self.paths.settings:
                if directory.is_symlink():
                    failures.append(f"{directory}: linked settings directory was not cleared")
                    continue
                targets.extend(
                    directory / name
                    for name in (
                        "settings.json",
                        "main-window.json",
                        "pipeline-presets.json",
                        "pipeline-selection.json",
                        "desktop.ini",
                        "maintenance.json",
                    )
                )
                for folder in ("camera-profiles", "dialog-directories"):
                    child = directory / folder
                    if child.is_dir() and not child.is_symlink():
                        if folder == "camera-profiles":
                            targets.extend(
                                path
                                for path in child.iterdir()
                                if re.fullmatch(r"[0-9a-f]{24}\.json", path.name)
                            )
                        else:
                            targets.extend(
                                child / f"{kind}.json"
                                for kind in (
                                    "image",
                                    "video",
                                    "timelapse",
                                    "temperatures",
                                    "pipeline_import",
                                    "pipeline_export",
                                )
                            )
        if "sessions" in self.categories:
            logs = self.paths.state / "debug-logs"
            if logs.is_dir() and not logs.is_symlink():
                targets.extend(
                    path
                    for path in logs.iterdir()
                    if re.fullmatch(
                        r"application\.log(?:\.[123])?|viewer(?:-\d{8}-\d{6}-\d{6})?\.(?:jsonl|stacks\.log|rejected-\d{2}\.bin)",
                        path.name,
                    )
                )
            baselines = self.paths.state / "camera-baselines"
            if baselines.is_dir() and not baselines.is_symlink():
                targets.extend(
                    path
                    for path in baselines.iterdir()
                    if re.fullmatch(r"[0-9a-f]{32}\.json", path.name)
                )
            targets.extend(self.paths.artifacts)
        if "runtimes" in self.categories and self.paths.runtimes.is_symlink():
            failures.append(f"{self.paths.runtimes}: linked runtime directory was not cleared")
        if (
            "runtimes" in self.categories
            and self.paths.runtimes.is_dir()
            and not self.paths.runtimes.is_symlink()
        ):
            targets.extend(
                path
                for path in self.paths.runtimes.iterdir()
                if re.fullmatch(
                    r"v1-(?:cpython|pypy)-\d+-(?:linux|darwin)-[^/]+-(?:cuda12|cuda13|cuda13-orin|coreml)",
                    path.name,
                )
            )
        if "outputs" in self.categories:
            for directory in self.paths.outputs:
                if directory.is_dir() and not directory.is_symlink():
                    targets.extend(
                        path
                        for path in directory.iterdir()
                        if path.is_file() and not path.is_symlink() and generated_output(path)
                    )
        if "bytecode" in self.categories:
            targets.extend(bytecode_files(self.paths.bytecode))
        for path in dict.fromkeys(targets):
            try:
                removed, bytes_removed = remove_target(path)
                count += removed
                size += bytes_removed
            except OSError as exc:
                failures.append(f"{path}: {exc}")
        if "models" in self.categories:
            try:
                removed, bytes_removed = clear_downloaded_models(
                    root=self.paths.models,
                    include_exports="settings" in self.categories,
                    clear_registry="settings" in self.categories,
                )
                count += removed
                size += bytes_removed
            except (OSError, ValueError) as exc:
                failures.append(f"{self.paths.models}: {exc}")
        if failures:
            raise OSError(
                "Maintenance incomplete; some items may already be removed: " + "; ".join(failures)
            )
        return count, size


def generated_output(path: Path) -> bool:
    """
    Identify app output filenames without deleting source folders or unrelated user files.
    """
    capture = re.fullmatch(
        r"(?:based-thermals|topdon-duo)-(?:video-|timelapse-|temperatures-)?\d{8}-\d{6}(?:[-_.]\w+)?\.(?:png|npz|json|mp4|csv)",
        path.name,
    )
    diagnostics = path.parent.name == "diagnostics" and path.suffix in (".jsonl", ".bin")
    return bool(capture or diagnostics or path.name.endswith(".pipeline.json"))


def bytecode_files(roots: tuple[Path, ...]) -> list[Path]:
    """
    Find generated Python caches without descending into dependencies, tooling or linked folders.
    """
    paths = []
    for root in roots:
        if root.is_symlink() or not root.is_dir():
            continue
        for directory, folders, files in os.walk(root, followlinks=False, onerror=walk_error):
            base = Path(directory)
            folders[:] = [
                name for name in folders if name not in SKIP and not (base / name).is_symlink()
            ]
            if "__pycache__" in folders:
                paths.append(base / "__pycache__")
                folders.remove("__pycache__")
            paths.extend(base / name for name in files if name.endswith((".pyc", ".pyo")))
    return paths


def remove_target(path: Path) -> tuple[int, int]:
    """
    Remove a scoped target, unlinking symlinks without touching their destination.
    """
    if path.is_symlink():
        path.unlink()
        return 1, 0
    if not path.exists():
        return 0, 0
    if path.is_dir():
        count, size = 0, 0
        for directory, folders, files in os.walk(path, followlinks=False, onerror=walk_error):
            base = Path(directory)
            count += len(files) + sum((base / name).is_symlink() for name in folders)
            size += sum(
                (base / name).stat().st_size for name in files if not (base / name).is_symlink()
            )
        shutil.rmtree(path)
        return count, size
    size = path.stat().st_size
    path.unlink()
    return 1, size


def walk_error(error: OSError) -> None:
    """
    Report unreadable cleanup locations rather than silently skipping their contents.
    """
    raise error
