"""
Protect backend ownership, public imports and the private helper launch target.
"""

import subprocess
import sys
from pathlib import Path

from based_thermals import camera, hardware_controls, queued_usb
from based_thermals.camera_backends import create_camera, registry
from based_thermals.camera_backends.duo import camera as duo_camera
from based_thermals.camera_backends.duo import hardware_controls as duo_controls
from based_thermals.camera_backends.duo import queued_usb as duo_queue
from based_thermals.camera_backends.duo.macos_usb.client import helper_command
from based_thermals.camera_backends.errors import CameraError
from based_thermals.camera_backends.virtual import VirtualDualCamera as LegacyVirtual
from based_thermals.camera_backends.virtual_dual import VirtualDualCamera


def test_public_imports_point_to_one_backend_implementation() -> None:
    """
    Preserve caller imports without duplicate camera/control/error classes or protocol state.
    """
    assert camera.TC002CDuoCamera is duo_camera.TC002CDuoCamera
    assert camera.decode_duo_frame is duo_camera.decode_duo_frame
    assert camera.CameraError is duo_camera.CameraError is CameraError
    assert hardware_controls.HardwareControls is duo_controls.HardwareControls
    assert queued_usb.QueuedBulkReader is duo_queue.QueuedBulkReader
    assert LegacyVirtual is VirtualDualCamera
    assert create_camera is registry.create_camera
    assert isinstance(create_camera("virtual-dual"), VirtualDualCamera)


def test_virtual_backend_and_shared_contracts_do_not_import_duo_transport(tmp_path: Path) -> None:
    """
    Import generic acquisition/control infrastructure in a fresh interpreter without Duo USB code.
    """
    script = """
import sys
from based_thermals.camera_backends import CameraProfile, create_camera
from based_thermals.camera_backends.controls import DeviceControls
from based_thermals.frame_pump import CameraFramePump
camera = create_camera('virtual-dual')
assert camera.profile.id == 'virtual-dual'
camera.create_controls()
for name in sys.modules:
    assert name not in ('based_thermals.camera', 'based_thermals.hardware_controls')
    assert not name.startswith('based_thermals.camera_backends.duo.camera')
    assert not name.startswith('based_thermals.camera_backends.duo.queued_usb')
    assert not name.startswith('based_thermals.camera_backends.duo.hardware_controls')
    assert not name.startswith('based_thermals.camera_backends.duo.macos_usb')
"""
    result = subprocess.run(
        [sys.executable, "-c", script], cwd=tmp_path, capture_output=True,
        text=True, timeout=30, check=False,
    )
    assert result.returncode == 0, result.stderr


def test_authorized_helper_uses_duo_owned_service() -> None:
    """
    Keep the private subprocess entry point aligned with the relocated Duo helper.
    """
    command = helper_command()
    assert command[-2:] == ["-m", "based_thermals.camera_backends.duo.macos_usb.service"]
