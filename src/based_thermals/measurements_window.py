"""
Measurements popup with persistent shapes, exclusion zones and label configuration.
"""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy

from PySide6.QtCore import QSignalBlocker, Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .capture_window import run_window
from .measurement_ui.appearance import AppearanceControls
from .measurement_ui.geometry import GeometryEditor
from .measurements.schema import (
    KINDS,
    LABEL_FIELDS,
    default_region,
    validate_center,
    validate_region,
)


class AllMeasurementsCheckBox(QCheckBox):
    """
    Display mixed visibility while making user clicks toggle only show-all/hide-all.
    """

    def nextCheckState(self) -> None:
        """
        Show every measurement from a mixed state, or hide every checked measurement.
        """
        state = (
            Qt.CheckState.Unchecked
            if self.checkState() == Qt.CheckState.Checked
            else Qt.CheckState.Checked
        )
        self.setCheckState(state)


class MeasurementsWindow(QWidget):
    """
    Edit measurement data in Qt while the parent owns sampling and locks.
    """

    def __init__(self, send: Callable[[dict], None]) -> None:
        """
        Build a measurement list, native geometry editor and configurable readout fields.
        """
        super().__init__()
        self.send = send
        self.setWindowTitle("Measurements")
        self.resize(980, 700)
        self._state: dict = {}
        self._items: dict[int, dict] = {}
        self._selected: int | None = None
        self._dirty = False
        self._loading = False
        self._creating = False
        self.sensor_size = (256, 192)
        self.region = default_region()
        self.center = (128, 96)
        layout = QVBoxLayout(self)
        self.show_all = AllMeasurementsCheckBox("Show all measurements")
        self.show_all.setTristate(True)
        self.show_all.setToolTip(
            "Enable or disable every measurement, including exclusion zones and their graphs."
        )
        self.show_all.stateChanged.connect(self._show_all)
        layout.addWidget(self.show_all)
        header = QHBoxLayout()
        self.kind = QComboBox()
        for kind in KINDS:
            self.kind.addItem("Bézier" if kind == "bezier" else kind.title(), kind)
        self.create = QPushButton("Create")
        self.create.clicked.connect(self._create)
        self.draw_exclude = QCheckBox("Exclusion")
        self.auto_hide = QCheckBox("Auto-hide")
        self.auto_hide.setChecked(True)
        self.auto_hide.setToolTip("Hide Measurements while drawing a new shape on the preview")
        self.placing = QCheckBox("Place spots on image (P)")
        self.placing.toggled.connect(
            lambda enabled: self.send({"action": "placing", "enabled": enabled})
        )
        self.dynamic_markers = QCheckBox("Dynamic markers")
        self.dynamic_markers.setToolTip(
            "Show 10 px circles at the coldest point (blue), hottest point (red), "
            "and camera center (green), with inverted contrast halos."
        )
        self.dynamic_markers.toggled.connect(
            lambda enabled: self.send({"action": "dynamic_markers", "enabled": enabled})
        )
        self.calibrate = QPushButton("Calibrate now")
        self.calibrate.clicked.connect(lambda: self.send({"action": "calibrate_now"}))
        header.addWidget(self.kind)
        header.addWidget(self.create)
        header.addWidget(self.draw_exclude)
        header.addWidget(self.auto_hide)
        header.addWidget(self.placing)
        header.addWidget(self.dynamic_markers)
        header.addStretch()
        header.addWidget(self.calibrate)
        layout.addLayout(header)
        split = QSplitter()
        left = QWidget()
        left_layout = QVBoxLayout(left)
        self.items = QListWidget()
        self.items.currentItemChanged.connect(self._select)
        self.items.itemChanged.connect(self._enabled)
        left_layout.addWidget(self.items)
        self.delete = QPushButton("Delete selected")
        self.delete.clicked.connect(self._delete)
        self.clear = QPushButton("Clear all measurements")
        self.clear.clicked.connect(lambda: self.send({"action": "clear_all"}))
        self.clear_history = QPushButton("Clear selected graph history")
        self.clear_history.clicked.connect(self._clear_history)
        for button in (self.delete, self.clear, self.clear_history):
            left_layout.addWidget(button)
        split.addWidget(left)
        self.editor_panel = QWidget()
        right = QVBoxLayout(self.editor_panel)
        form = QFormLayout()
        self.name = QLineEdit()
        self.name.setMaxLength(64)
        self.name.textEdited.connect(self._mark_dirty)
        self.shape = QLabel()
        self.z_index = QSpinBox()
        self.z_index.setRange(-1000, 1000)
        self.z_index.setKeyboardTracking(False)
        self.z_index.setToolTip(
            "Higher values draw above lower values. Equal values keep their order."
        )
        self.z_index.valueChanged.connect(self._mark_dirty)
        z_editor = self.z_index.lineEdit()
        assert z_editor is not None
        z_editor.textEdited.connect(self._mark_dirty)
        self.bring_front = QPushButton("Bring to front")
        self.send_back = QPushButton("Send to back")
        self.bring_front.clicked.connect(lambda: self._shift_z_index(front=True))
        self.send_back.clicked.connect(lambda: self._shift_z_index(front=False))
        z_controls = QHBoxLayout()
        z_controls.addWidget(self.z_index)
        z_controls.addWidget(self.bring_front)
        z_controls.addWidget(self.send_back)
        self.radius = self._spin(0, 256, 0.5)
        self.diameter = self._spin(0, 512, 1)
        self.radius.valueChanged.connect(self._radius_changed)
        self.diameter.valueChanged.connect(self._diameter_changed)
        self.x = self._spin(0, 255, 1)
        self.y = self._spin(0, 191, 1)
        self.x.setDecimals(0)
        self.y.setDecimals(0)
        self.x.valueChanged.connect(self._center_changed)
        self.y.valueChanged.connect(self._center_changed)
        self.exclude = QCheckBox("Exclusion zone: subtract from measurements; no graph")
        self.exclude.toggled.connect(self._exclusion_changed)
        form.addRow("Name", self.name)
        form.addRow("Shape", self.shape)
        form.addRow("Z index", z_controls)
        form.addRow("Spot radius (native pixels)", self.radius)
        form.addRow("Spot diameter (native pixels)", self.diameter)
        coordinates = QHBoxLayout()
        coordinates.addWidget(self.x)
        coordinates.addWidget(self.y)
        form.addRow("Spot position (x, y)", coordinates)
        form.addRow(self.exclude)
        right.addLayout(form)
        self.appearance = AppearanceControls(self._appearance_changed, self._mark_dirty)
        right.addWidget(self.appearance)
        labels = QHBoxLayout()
        labels.addWidget(QLabel("Label fields:"))
        self.label_fields: dict[str, QCheckBox] = {}
        for field in LABEL_FIELDS:
            control = QCheckBox(field.title())
            control.toggled.connect(self._mark_dirty)
            labels.addWidget(control)
            self.label_fields[field] = control
        right.addLayout(labels)
        right.addWidget(
            QLabel("Uncheck every field to hide the label. Area labels attach to the outside edge.")
        )
        self.preview = GeometryEditor(self._geometry_changed)
        right.addWidget(self.preview, 1)
        self.points = QTableWidget(0, 3)
        self.points.setHorizontalHeaderLabels(("Vertex / handle", "X (px)", "Y (px)"))
        self.points.setMaximumHeight(150)
        self.points.itemChanged.connect(self._table_changed)
        self.points.currentCellChanged.connect(self._select_handle)
        right.addWidget(self.points)
        point_buttons = QHBoxLayout()
        self.add_vertex = QPushButton("Add vertex / Bézier segment")
        self.remove_vertex = QPushButton("Remove selected vertex / segment")
        self.add_vertex.clicked.connect(self._add_vertex)
        self.remove_vertex.clicked.connect(self._remove_vertex)
        point_buttons.addWidget(self.add_vertex)
        point_buttons.addWidget(self.remove_vertex)
        right.addLayout(point_buttons)
        self.apply = QPushButton("Apply configuration")
        self.apply.clicked.connect(self._apply)
        split.addWidget(self.editor_panel)
        split.setSizes([240, 700])
        layout.addWidget(split, 1)
        actions = QHBoxLayout()
        actions.addStretch()
        actions.addWidget(self.apply)
        self.close_button = QPushButton("Close")
        self.close_button.clicked.connect(self.close)
        actions.addWidget(self.close_button)
        layout.addLayout(actions)
        self.status = QLabel(
            "Create a shape, then edit its coordinates or drag its handles in the grid."
        )
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.update_state({})

    @staticmethod
    def _spin(low: float, high: float, step: float) -> QDoubleSpinBox:
        """
        Build a numeric control that commits edits without emitting every keystroke.
        """
        control = QDoubleSpinBox()
        control.setRange(low, high)
        control.setDecimals(2)
        control.setSingleStep(step)
        control.setKeyboardTracking(False)
        return control

    def _show_all(self, state: int) -> None:
        """
        Send one bulk command without changing pending geometry or label edits.
        """
        if self._items and not self._state.get("locked"):
            self.send({"action": "enable_all", "enabled": state == Qt.CheckState.Checked.value})

    def _mark_dirty(self, *_args: object) -> None:
        """
        Keep unsaved geometry intact when live parent state arrives.
        """
        if not self._loading:
            self._dirty = True

    def _create(self) -> None:
        """
        Arm the selected video drawing tool and uncover the camera image.
        """
        if self._state.get("locked"):
            return
        if self._dirty and not self._apply():
            return
        self.send(
            {
                "action": "draw",
                "kind": self.kind.currentData(),
                "exclude": self.draw_exclude.isChecked(),
                "orientation": self._state.get("orientation"),
            }
        )
        self._creating = True
        if self.auto_hide.isChecked():
            self.hide()

    def _delete(self) -> None:
        """
        Delete the selected measurement through the parent lock boundary.
        """
        if self._selected is not None and not self._state.get("locked"):
            self.send({"action": "clear", "spot": self._selected})
            self._dirty = False

    def _clear_history(self) -> None:
        """
        Clear samples while retaining the selected shape and its settings.
        """
        if self._selected is not None and not self._state.get("locked"):
            self.send({"action": "clear_history", "spot": self._selected})

    def _enabled(self, item: QListWidgetItem) -> None:
        """
        Enable/disable a measurement without deleting its configuration.
        """
        if not self._loading and not self._state.get("locked"):
            self.send(
                {
                    "action": "enable",
                    "spot": item.data(Qt.ItemDataRole.UserRole),
                    "enabled": item.checkState() == Qt.CheckState.Checked,
                }
            )

    def _select(self, current: QListWidgetItem | None, _previous: QListWidgetItem | None) -> None:
        """
        Commit valid edits before switching the active configuration form.
        """
        if self._loading:
            return
        if self._dirty and not self._apply():
            with QSignalBlocker(self.items):
                for index in range(self.items.count()):
                    item = self.items.item(index)
                    if item.data(Qt.ItemDataRole.UserRole) == self._selected:
                        self.items.setCurrentItem(item)
            return
        self._selected = current.data(Qt.ItemDataRole.UserRole) if current else None
        self._load_selected()

    def _load_selected(self) -> None:
        """
        Populate controls without generating commands or discarding live edits.
        """
        selected = self._items.get(self._selected) if self._selected is not None else None
        self.editor_panel.setEnabled(selected is not None and not self._state.get("locked"))
        if selected is None:
            self._dirty = False
            return
        self._loading = True
        self.region = deepcopy(selected["region"])
        self.center = selected["pixel"][0], selected["pixel"][1]
        self.name.setText(selected["name"])
        self.shape.setText(self.region["kind"].title())
        self.z_index.setValue(self.region["z_index"])
        self.radius.setMaximum(max(self.sensor_size))
        self.diameter.setMaximum(max(self.sensor_size) * 2)
        self.radius.setValue(self.region["radius"])
        self.diameter.setValue(self.region["radius"] * 2)
        self.x.setMaximum(self.sensor_size[0] - 1)
        self.y.setMaximum(self.sensor_size[1] - 1)
        self.x.setValue(self.center[0])
        self.y.setValue(self.center[1])
        for control in (self.radius, self.diameter, self.x, self.y):
            control.setEnabled(self.region["kind"] == "spot")
        self.exclude.setChecked(self.region["exclude"])
        self._update_label_visibility()
        self.appearance.set_region(self.region)
        for field, control in self.label_fields.items():
            control.setChecked(field in self.region["labels"])
        self.preview.set_geometry(self.region, self.center, self.sensor_size)
        self._fill_points()
        self.add_vertex.setEnabled(self.region["kind"] in ("polygon", "bezier"))
        self.remove_vertex.setEnabled(self.region["kind"] in ("polygon", "bezier"))
        self._loading = False
        self._dirty = False

    def _fill_points(self) -> None:
        """
        Show each polygon vertex or explicitly named Bezier anchor/control handle.
        """
        with QSignalBlocker(self.points):
            self.points.setRowCount(len(self.region["points"]))
            for row, point in enumerate(self.region["points"]):
                role = (
                    ("Anchor", "Control 1", "Control 2")[row % 3]
                    if self.region["kind"] == "bezier"
                    else "Vertex"
                )
                heading = QTableWidgetItem(f"{row + 1}: {role}")
                heading.setFlags(heading.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self.points.setItem(row, 0, heading)
                for axis in (0, 1):
                    self.points.setItem(row, axis + 1, QTableWidgetItem(f"{point[axis]:g}"))

    def _table_changed(self, _item: QTableWidgetItem) -> None:
        """
        Validate numeric coordinate edits before updating the geometry preview.
        """
        if self._loading:
            return
        try:
            points = self._read_points()
            region = validate_region({**self.region, "points": points}, self.sensor_size)
        except (ValueError, TypeError) as exc:
            self.status.setText(str(exc))
            self._dirty = True
            return
        self._geometry_changed(region, self.center, refresh_table=False)

    def _read_points(self) -> list[list[float]]:
        """
        Read a complete numeric coordinate table, rejecting empty/incomplete cells.
        """
        points = []
        for row in range(self.points.rowCount()):
            coordinates = []
            for column in (1, 2):
                cell = self.points.item(row, column)
                if cell is None:
                    raise ValueError("Every geometry point needs x and y coordinates")
                coordinates.append(float(cell.text()))
            points.append(coordinates)
        return points

    def _select_handle(
        self, row: int, _column: int, _previous_row: int, _previous_column: int
    ) -> None:
        """
        Highlight the coordinate row's handle in the shape preview.
        """
        self.preview.selected = row
        self.preview.update()

    def _geometry_changed(
        self, region: dict, center: tuple[int, int], *, refresh_table: bool = True
    ) -> None:
        """
        Synchronize dragged handles, radius and center edits with the form.
        """
        self.region, self.center = region, center
        with QSignalBlocker(self.x), QSignalBlocker(self.y):
            self.x.setValue(center[0])
            self.y.setValue(center[1])
        if refresh_table:
            self._fill_points()
        self.preview.set_geometry(self.region, self.center, self.sensor_size)
        self._mark_dirty()

    def _radius_changed(self, value: float) -> None:
        """
        Keep radius and diameter equivalent in native pixels.
        """
        if not self._loading:
            with QSignalBlocker(self.diameter):
                self.diameter.setValue(value * 2)
            self._geometry_changed({**self.region, "radius": value}, self.center)

    def _diameter_changed(self, value: float) -> None:
        """
        Express a diameter edit as the same circular sampling radius.
        """
        if not self._loading:
            self.radius.setValue(value / 2)

    def _center_changed(self, _value: float) -> None:
        """
        Reposition a spot in native-grid coordinates.
        """
        if not self._loading:
            self._geometry_changed(self.region, (int(self.x.value()), int(self.y.value())))

    def _exclusion_changed(self, value: bool) -> None:
        """
        Show exclusion styling without calculating temperatures in Qt.
        """
        if not self._loading:
            self._geometry_changed({**self.region, "exclude": value}, self.center)
        self._update_label_visibility()

    def _update_label_visibility(self) -> None:
        """
        Hide temperature label choices for exclusions, retaining their saved checks.
        """
        for field in ("min", "max", "average"):
            self.label_fields[field].setVisible(not self.exclude.isChecked())

    def _shift_z_index(self, *, front: bool) -> None:
        """
        Move the pending selected layer beyond other saved measurements' Z indices.
        """
        if self._selected is None or self._state.get("locked"):
            return
        self.z_index.interpretText()
        others = [
            entry["region"]["z_index"]
            for number, entry in self._items.items()
            if number != self._selected
        ]
        if not others:
            return
        current = self.z_index.value()
        target = max(current, max(others) + 1) if front else min(current, min(others) - 1)
        if not self.z_index.minimum() <= target <= self.z_index.maximum():
            self.status.setText("Z index limit reached; adjust the other measurement layers first.")
            return
        self.z_index.setValue(target)

    def _appearance_changed(self, values: dict) -> None:
        """
        Preview display-only styling through the same local edit boundary as geometry.
        """
        if not self._loading and not self._state.get("locked"):
            self._geometry_changed({**self.region, **values}, self.center, refresh_table=False)

    def _add_vertex(self) -> None:
        """
        Add a polygon vertex or a full anchor/control/control cubic segment.
        """
        if self._state.get("locked") or len(self.region["points"]) >= 192:
            return
        count = 3 if self.region["kind"] == "bezier" else 1
        region = deepcopy(self.region)
        region["points"].extend([list(self.center) for _ in range(count)])
        self._geometry_changed(region, self.center)

    def _remove_vertex(self) -> None:
        """
        Preserve the minimum topology when removing a selected vertex/segment.
        """
        if self._state.get("locked"):
            return
        row = self.points.currentRow()
        count, minimum = (3, 6) if self.region["kind"] == "bezier" else (1, 3)
        if row < 0 or len(self.region["points"]) <= minimum:
            return
        region = deepcopy(self.region)
        index = row // count * count
        del region["points"][index : index + count]
        self._geometry_changed(region, self.center)

    def _apply(self) -> bool:
        """
        Commit validated geometry and label settings through the parent process.
        """
        if self._selected is None or self._state.get("locked"):
            return False
        for control in (self.radius, self.diameter, self.x, self.y, self.z_index):
            control.interpretText()
        if self.points.rowCount():
            try:
                self.region["points"] = self._read_points()
            except ValueError:
                self.status.setText("Geometry coordinates must be numbers.")
                return False
        try:
            region = validate_region(
                {
                    **self.region,
                    **self.appearance.values(),
                    "z_index": self.z_index.value(),
                    "labels": [
                        field for field, control in self.label_fields.items() if control.isChecked()
                    ],
                },
                self.sensor_size,
            )
        except (ValueError, TypeError) as exc:
            self.status.setText(str(exc))
            return False
        self.send(
            {
                "action": "configure",
                "spot": self._selected,
                "pixel": list(self.center),
                "region": region,
                "orientation": self._state.get("orientation"),
            }
        )
        self.send({"action": "rename", "spot": self._selected, "name": self.name.text()})
        self._items[self._selected].update(
            region=deepcopy(region), pixel=list(self.center), name=self.name.text()
        )
        self._dirty = False
        self.status.setText("Configuration applied.")
        return True

    def update_state(self, state: dict) -> None:
        """
        Update locks/list state without replacing focused configuration edits.
        """
        size = state.get("sensor_size", self.sensor_size)
        if (
            not isinstance(size, (list, tuple))
            or len(size) != 2
            or any(type(n) is not int or n < 1 for n in size)
        ):
            self.status.setText("Invalid native sensor dimensions")
            return
        sensor_size = size[0], size[1]
        entries = state.get("spots", [])
        try:
            if not isinstance(entries, list) or len(entries) > 1000:
                raise ValueError("Invalid measurement list")
            validated = []
            numbers = set()
            for entry in entries:
                if not isinstance(entry, dict):
                    raise TypeError("Invalid measurement entry")
                number, name, enabled = entry.get("number"), entry.get("name"), entry.get("enabled")
                if (
                    not isinstance(number, int)
                    or isinstance(number, bool)
                    or number < 1
                    or number in numbers
                    or not isinstance(name, str)
                    or len(name) > 64
                    or type(enabled) is not bool
                ):
                    raise ValueError("Invalid measurement identity or enabled state")
                center = validate_center(entry.get("pixel"), sensor_size)
                region = validate_region(entry.get("region"), sensor_size)
                validated.append(
                    {
                        "number": number,
                        "name": name,
                        "enabled": enabled,
                        "pixel": list(center),
                        "region": region,
                    }
                )
                numbers.add(number)
        except (TypeError, ValueError) as exc:
            self.status.setText(str(exc))
            return
        changed_view = state.get("orientation") != self._state.get("orientation")
        if state.get("drawing_cancelled"):
            self._creating = False
        self._state = state
        self.sensor_size = sensor_size
        entries = validated
        new_items = {entry["number"]: entry for entry in entries}
        changed_ids = list(new_items) != list(self._items)
        self._items = deepcopy(new_items)
        locked = bool(state.get("locked"))
        enabled_count = sum(entry["enabled"] for entry in entries)
        all_state = (
            Qt.CheckState.Unchecked
            if enabled_count == 0
            else Qt.CheckState.Checked
            if enabled_count == len(entries)
            else Qt.CheckState.PartiallyChecked
        )
        with QSignalBlocker(self.show_all):
            self.show_all.setCheckState(all_state)
            self.show_all.setEnabled(bool(entries) and not locked)
        self._loading = True
        if changed_ids:
            with QSignalBlocker(self.items):
                self.items.clear()
                for entry in entries:
                    item = QListWidgetItem()
                    item.setData(Qt.ItemDataRole.UserRole, entry["number"])
                    self.items.addItem(item)
                if self._creating and entries:
                    self._selected = entries[-1]["number"]
                    self._creating = False
                elif self._selected not in new_items:
                    self._selected = entries[0]["number"] if entries else None
        for row, entry in enumerate(entries):
            item = self.items.item(row)
            with QSignalBlocker(self.items):
                item.setText(
                    f"{entry['number']}: {entry['name']} · {entry['region']['kind']}"
                    + (" · exclusion" if entry["region"]["exclude"] else "")
                )
                flags = Qt.ItemFlag(
                    Qt.ItemFlag.ItemIsEnabled.value
                    | Qt.ItemFlag.ItemIsSelectable.value
                    | (Qt.ItemFlag.ItemIsUserCheckable.value if not locked else 0)
                )
                item.setFlags(flags)
                item.setCheckState(
                    Qt.CheckState.Checked if entry["enabled"] else Qt.CheckState.Unchecked
                )
                if entry["number"] == self._selected:
                    self.items.setCurrentItem(item)
        with QSignalBlocker(self.placing):
            self.placing.setChecked(bool(state.get("placing")))
        with QSignalBlocker(self.dynamic_markers):
            self.dynamic_markers.setChecked(bool(state.get("dynamic_markers", False)))
        self._loading = False
        for control in (self.kind, self.create, self.draw_exclude, self.placing, self.clear):
            control.setEnabled(not locked)
        for control in (self.create, self.placing):
            control.setEnabled(not locked and state.get("mapping_available", True))
        self.dynamic_markers.setEnabled(not locked and state.get("mapping_available", True))
        selected = self._selected in new_items
        self.delete.setEnabled(selected and not locked)
        self.clear_history.setEnabled(
            selected
            and not locked
            and not (new_items.get(self._selected, {}).get("region", {}).get("exclude", False))
        )
        self.calibrate.setEnabled(not locked and state.get("calibration_available", False))
        self.editor_panel.setEnabled(selected and not locked)
        self.apply.setEnabled(selected and not locked)
        if changed_ids or changed_view or locked or not self._dirty:
            self._load_selected()
        if locked:
            self.status.setText("Stop logging or close calibration to edit measurements.")
        if state.get("raise_window"):
            self.showNormal()
            self.raise_()
            self.activateWindow()


def main() -> int:
    """
    Run measurement configuration outside the acquisition process.
    """
    result = run_window(MeasurementsWindow, "Measurements")
    return result


if __name__ == "__main__":
    raise SystemExit(main())
