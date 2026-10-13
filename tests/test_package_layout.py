"""
Verify the installed application exposes the renamed package and launch commands.
"""

import importlib.util
import subprocess
import sys
from importlib.metadata import distribution
from importlib.resources import files
from pathlib import Path

import pytest


def test_application_package_and_resources() -> None:
    """
    Keep the implementation and bundled resources in the sole application namespace.
    """
    assert importlib.util.find_spec("topdon_duo") is None
    resources = files("based_thermals")
    assert resources.joinpath("predefined_pipeline_presets.json").is_file()
    assert resources.joinpath("models", "acnet-legacy-hdn0.onnx").is_file()


@pytest.mark.parametrize(
    "command",
    (
        "based-thermals",
        "based-thermals-desktop",
        "based-thermals-models",
    ),
)
def test_installed_commands_launch_renamed_package(command: str, tmp_path: Path) -> None:
    """
    Launch installed commands outside the checkout without accessing a camera.
    """
    application = distribution("based-thermals")
    entry_point = next(point for point in application.entry_points if point.name == command)
    assert entry_point.value.startswith("based_thermals.")
    executable = Path(sys.executable).parent / command
    result = subprocess.run(
        [str(executable), "--help"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout.lower()
