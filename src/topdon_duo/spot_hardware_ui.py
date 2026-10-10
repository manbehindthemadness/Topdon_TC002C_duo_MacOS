"""
Camera-native spot corrections, independent of viewer sampling spots and pipelines.
"""

from typing import Any

from PySide6.QtCore import QSignalBlocker
from PySide6.QtWidgets import QVBoxLayout, QWidget

from .pipeline_ui.hardware import row_options
from .view_controls import ControlRow, NoWheelComboBox


class SpotHardwareControls(QWidget):
    """
    Present backend-reported spot identities and supported correction fields.
    """

    def __init__(self, send: Any) -> None:
        """
        Create wheel-safe fields without reading or writing the camera.
        """
        super().__init__()
        self.send = send
        self.rows: dict[str, ControlRow] = {}
        self.state: dict[str, Any] = {}
        self.locked = False
        self.signature = None
        self.layout_rows = QVBoxLayout(self)
        self.selector = NoWheelComboBox()
        self.selector.setAccessibleName("Camera measurement spot")
        self.selector.currentIndexChanged.connect(self.refresh)
        self.layout_rows.addWidget(self.selector)

    def update_state(self, state: dict[str, Any], locked: bool) -> None:
        """
        Reconcile metadata and readbacks without publishing changes from state updates.
        """
        self.state, self.locked = state, locked
        spots = state.get("spot_hardware", {})
        specs = {name: spec for name, spec in state.get("camera_capabilities", {}).get(
            "controls", {},
        ).items() if spec.get("scope") == "spot" and spec.get("effect") == "measurement"
                 and spec.get("supported")}
        signature = list(spots), specs
        if signature != self.signature:
            selected = self.selector.currentData()
            with QSignalBlocker(self.selector):
                self.selector.clear()
                for spot_id in spots:
                    self.selector.addItem(f"Camera spot {spot_id}", spot_id)
                self.selector.setCurrentIndex(max(0, self.selector.findData(selected)))
            for row in self.rows.values():
                row.timer.stop()
                self.layout_rows.removeWidget(row)
                row.deleteLater()
            self.rows = {}
            for name, spec in specs.items():
                row = ControlRow(spec["title"], lambda value, field=name: self.change(field, value),
                                 **row_options(spec))
                self.layout_rows.addWidget(row)
                self.rows[name] = row
            self.signature = signature
        self.setVisible(bool(spots and specs))
        self.selector.setEnabled(not locked)
        self.refresh()

    def refresh(self) -> None:
        """
        Display current readbacks while retaining the selected camera spot.
        """
        spot = self.state.get("spot_hardware", {}).get(self.selector.currentData(), {})
        for name, row in self.rows.items():
            current = spot.get(name, {})
            row.set_display_unit(self.state.get("temperature_unit", "C"))
            row.update_state(current.get("value", 0), not self.locked and current.get("available"))

    def change(self, name: str, value: object) -> None:
        """
        Send an explicit spot-scoped setting only for an unlocked available field.
        """
        spot_id = self.selector.currentData()
        current = self.state.get("spot_hardware", {}).get(spot_id, {}).get(name, {})
        if not self.locked and current.get("available"):
            self.send({"action": "hardware", "spot_id": spot_id, "name": name,
                       "value": value, "enabled": True})
