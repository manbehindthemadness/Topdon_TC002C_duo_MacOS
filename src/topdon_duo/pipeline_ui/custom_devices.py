"""
Custom execution and Apple compute-device controls using viewer capability state.
"""

from collections.abc import Callable
from typing import Any

from PySide6.QtCore import QSignalBlocker
from PySide6.QtGui import QStandardItemModel
from PySide6.QtWidgets import QComboBox, QFormLayout, QLabel, QVBoxLayout, QWidget

from ..custom_nodes.devices import APPLE_COMPUTE, BACKENDS, resolve_device, validate_device


class DevicePicker(QWidget):
    """
    Display effective CPU/GPU execution while retaining a portable saved preference.
    """

    def __init__(
        self, value: object, changed: Callable[[dict[str, str]], None],
        state: dict[str, Any] | None = None,
    ) -> None:
        """
        Create device choices without probing or creating inference sessions in the UI.
        """
        super().__init__()
        self.changed = changed
        self.state = state or {}
        self.value = validate_device(value)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        form = QFormLayout()
        self.backend = QComboBox()
        self.backend.setAccessibleName("Execution")
        self.compute = QComboBox()
        self.compute.setAccessibleName("Compute devices")
        for key, label in BACKENDS:
            self.backend.addItem(label, key)
        for key, label in APPLE_COMPUTE:
            self.compute.addItem(label, key)
        self.compute_label = QLabel("Compute devices")
        form.addRow("Execution", self.backend)
        form.addRow(self.compute_label, self.compute)
        layout.addLayout(form)
        self.status = QLabel()
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.refresh(value)
        self.backend.currentIndexChanged.connect(self.change_backend)
        self.compute.currentIndexChanged.connect(self.change_compute)

    def refresh(self, value: object, state: dict[str, Any] | None = None) -> None:
        """
        Refresh effective choices silently, preserving unavailable saved GPU settings.
        """
        self.value = validate_device(value)
        if state is not None:
            self.state = state
        apple = bool(self.state.get("apple_acceleration", {}).get("available"))
        nvidia = bool(self.state.get("nvidia_acceleration", {}).get("available"))
        effective = resolve_device(self.value, apple_available=apple, nvidia_available=nvidia)
        with QSignalBlocker(self.backend), QSignalBlocker(self.compute):
            self.backend.setCurrentIndex(self.backend.findData(effective["backend"]))
            self.compute.setCurrentIndex(self.compute.findData(effective["apple_compute"]))
        model = self.backend.model()
        if isinstance(model, QStandardItemModel):
            for index in range(self.backend.count()):
                selected = self.backend.itemData(index)
                item = model.item(index)
                if item is not None:
                    item.setEnabled(
                        selected == "cpu" or selected == "coreml" and apple
                        or selected == "cuda" and nvidia
                    )
        self.backend.setEnabled(apple or nvidia)
        self.compute.setEnabled(effective["backend"] == "coreml")
        show_compute = apple and effective["backend"] != "cuda"
        self.compute.setVisible(show_compute)
        self.compute_label.setVisible(show_compute)
        fallback = effective["backend"] != self.value["backend"]
        label = dict(BACKENDS)[effective["backend"]]
        self.status.setText(f"{label} execution · saved GPU settings retained" if fallback else "")
        self.status.setVisible(fallback)
        self.setToolTip(
            "Unavailable saved GPU preferences try the other GPU before CPU and remain saved."
        )

    def change_backend(self, _index: int) -> None:
        """
        Publish execution edits without replacing the saved Apple compute preference.
        """
        if not self.backend.isEnabled():
            self.refresh(self.value)
            return
        value = {**self.value, "backend": self.backend.currentData()}
        effective = resolve_device(
            value,
            apple_available=bool(self.state.get("apple_acceleration", {}).get("available")),
            nvidia_available=bool(self.state.get("nvidia_acceleration", {}).get("available")),
        )
        if effective["backend"] != value["backend"]:
            self.refresh(self.value)
            return
        self.refresh(value)
        self.changed(value)

    def change_compute(self, _index: int) -> None:
        """
        Publish Apple compute edits only when the control is available and unlocked.
        """
        if not self.compute.isEnabled():
            self.refresh(self.value)
            return
        value = {**self.value, "apple_compute": self.compute.currentData()}
        self.refresh(value)
        self.changed(value)
