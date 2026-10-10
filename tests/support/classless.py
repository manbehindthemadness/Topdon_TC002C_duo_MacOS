"""
Shared isolated package fixtures for the Classless YOLO example.
"""

import sys
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType

import pytest

from topdon_duo.custom_nodes.bundle import load_folder
from topdon_duo.custom_nodes.runtime import CustomPackage

FOLDER = Path(__file__).resolve().parents[2] / "examples/custom_nodes/Classless YOLO"


@pytest.fixture
def package() -> Iterator[CustomPackage]:
    """
    Import the portable example under a temporary namespace without requesting weights.
    """
    loaded = CustomPackage(load_folder(FOLDER)["package"])
    try:
        yield loaded
    finally:
        loaded.close()


def helper(loaded: CustomPackage, name: str) -> ModuleType:
    """
    Retrieve a relative helper module independently of the package's display name.
    """
    return sys.modules[loaded.name + "." + name]
