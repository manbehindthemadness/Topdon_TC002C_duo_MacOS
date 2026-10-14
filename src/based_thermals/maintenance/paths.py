"""
Explicit application-owned maintenance locations; source and third-party data stay separate.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

from ..dialog_preferences import load_dialog_directory
from ..onnx_models import model_path
from ..runtime_setup import cache_root
from ..window_preferences import _path


@dataclass(frozen=True)
class MaintenancePaths:
    """
    Bound cleanup to app storage, generated artifacts and Python bytecode roots.
    """

    settings: tuple[Path, ...]
    models: Path
    runtimes: Path
    state: Path
    bytecode: tuple[Path, ...]
    artifacts: tuple[Path, ...]
    outputs: tuple[Path, ...]

    @classmethod
    def discover(cls, output: Path | None = None) -> MaintenancePaths:
        """
        Resolve platform/override storage without scanning unrelated home directories.
        """
        package = Path(__file__).resolve().parent.parent
        project = package.parent.parent if package.parent.name == "src" else None
        checkout = (
            project if project is not None and (project / "pyproject.toml").is_file() else None
        )
        working = Path.cwd()
        directories = tuple(dict.fromkeys(path for path in (checkout, working) if path is not None))
        settings = [_path().parent]
        if sys.platform == "darwin":
            settings.append(Path.home() / "Library/Preferences/based-thermals")
        state = (
            Path(os.environ.get("XDG_STATE_HOME", Path.home() / ".local/state")) / "based-thermals"
        )
        bytecode = (checkout,) if checkout is not None else (package,)
        if sys.pycache_prefix:
            prefix = Path(sys.pycache_prefix)
            bytecode += tuple(prefix / str(root).lstrip(os.sep) for root in bytecode)
        artifacts = tuple(
            directory / name
            for directory in directories
            for name in (
                ":memory:.ses",
                ":memory:.ses-wal",
                ":memory:.ses-shm",
                ".pytest_cache",
                ".ruff_cache",
            )
        )
        outputs = tuple(
            dict.fromkeys(
                [
                    *(
                        directory / name
                        for directory in directories
                        for name in ("captures", "diagnostics")
                    ),
                    *directories,
                    *(
                        path
                        for kind in (
                            "image",
                            "video",
                            "timelapse",
                            "temperatures",
                            "pipeline_export",
                        )
                        if (path := load_dialog_directory(kind)) is not None
                    ),
                    *((output.expanduser(),) if output is not None else ()),
                ]
            )
        )
        return cls(
            tuple(dict.fromkeys(settings)),
            model_path("espcn").parent,
            cache_root(),
            state,
            bytecode,
            artifacts,
            outputs,
        )

    def descriptions(self) -> dict[str, list[str]]:
        """
        Present the concrete cleanup scope to the isolated confirmation dialog.
        """
        result = {
            "models": [str(self.models)],
            "runtimes": [str(self.runtimes)],
            "settings": [*map(str, self.settings), str(self.models / "custom-models.json")],
            "sessions": [str(self.state), *map(str, self.artifacts)],
            "bytecode": [str(path) for path in self.bytecode],
            "outputs": [str(path) for path in self.outputs],
        }
        return result
