"""
Execute cached custom packages in the processing worker, with checked image outputs.
"""

from __future__ import annotations

import importlib.util
import sys
from collections.abc import Callable
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, cast
from uuid import uuid4

import numpy as np

from ..enhancement_limits import MAX_PIXELS
from .bundle import package_files, read_json_object
from .configuration import read_controls
from .devices import resolve_device
from .labels import DisplayLabel, collect_labels


class CustomPackage:
    """
    Own one isolated module namespace and its temporary package directory.
    """

    process: Callable[[np.ndarray, dict[str, Any]], np.ndarray]

    def __init__(self, source: str) -> None:
        """
        Materialize a validated package; import it only in the processing worker.
        """
        self.source = source
        self.name = f"_topdon_custom_{uuid4().hex}"
        self.directory = TemporaryDirectory(prefix="topdon-custom-")
        try:
            root = Path(self.directory.name)
            for name, content in package_files(source).items():
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content, encoding="utf-8")
            spec = importlib.util.spec_from_file_location(
                self.name, root / "__init__.py", submodule_search_locations=[str(root)]
            )
            if spec is None or spec.loader is None:
                raise ValueError("Could not load custom package")
            module = importlib.util.module_from_spec(spec)
            sys.modules[self.name] = module
            spec.loader.exec_module(module)
            callback = getattr(module, "process", None)
            if not callable(callback):
                raise TypeError("Custom __init__.py must expose process(image, config)")
            self.process = cast(Callable[[np.ndarray, dict[str, Any]], np.ndarray], callback)
        except (Exception, SystemExit):
            # User packages can raise arbitrary exceptions at import; release every resource.
            self.close()
            raise

    def close(self) -> None:
        """
        Remove the package and relative imports, then release its temporary files.
        """
        for name in tuple(sys.modules):
            if name == self.name or name.startswith(self.name + "."):
                sys.modules.pop(name, None)
        self.directory.cleanup()


class CustomProcessor:
    """
    Cache package imports per node while preserving independent branch state.
    """

    def __init__(
        self, apple_available: bool | None = None, nvidia_available: bool | None = None,
    ) -> None:
        """
        Start with no loaded custom packages.
        """
        self.packages: dict[str, CustomPackage] = {}
        self.labels: list[DisplayLabel] = []
        self.apple_available = apple_available
        self.nvidia_available = nvidia_available

    def retain(self, identities: set[str]) -> None:
        """
        Unload removed or bypassed nodes.
        """
        for identity in tuple(self.packages):
            if identity not in identities:
                self.packages.pop(identity).close()

    def close(self) -> None:
        """
        Release all custom module resources.
        """
        self.retain(set())

    def apply(self, image: np.ndarray, item: dict[str, Any]) -> np.ndarray:
        """
        Run trusted user code on a copy and validate its display-only result.

        Arbitrary package exceptions become pipeline errors, retaining the viewer's
        existing last-valid-frame behavior. Custom code runs with application privileges.
        """
        params, identity = item["params"], item["id"]
        if params["package"] == "{}":
            previous = self.packages.pop(identity, None)
            if previous is not None:
                previous.close()
            return image
        try:
            package = self.packages.get(identity)
            if package is None or package.source != params["package"]:
                if package is not None:
                    self.packages.pop(identity).close()
                package = CustomPackage(params["package"])
                self.packages[identity] = package
            config = read_json_object(params["config"])
            for control in read_controls(params.get("controls", "[]"), config):
                if control["type"] == "device":
                    key = control["key"]
                    config[key] = resolve_device(
                        config[key], apple_available=self.apple_available,
                        nvidia_available=self.nvidia_available,
                    )
            with collect_labels(self.labels):
                result = package.process(image.copy(), config)
            if (
                not isinstance(result, np.ndarray)
                or result.ndim != 3
                or result.shape[2] != 3
                or min(result.shape[:2]) < 1
                or result.shape[0] * result.shape[1] > MAX_PIXELS
                or result.dtype.kind not in "uif"
                or not np.isfinite(result).all()
            ):
                raise ValueError("Return a finite H×W×3 numeric image within the 4 megapixel limit")
            output = np.clip(result, 0, 255).astype(np.float32)
        except (Exception, SystemExit) as exc:
            # User callbacks are an intentional fail-soft boundary for arbitrary Python errors.
            raise ValueError(
                f"Custom {params['name'] or 'module'}: {type(exc).__name__}: {exc}"
            ) from exc
        return output
