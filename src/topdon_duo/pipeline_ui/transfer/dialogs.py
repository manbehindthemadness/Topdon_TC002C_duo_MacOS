"""
Pipeline checklists with explicit import name-conflict choices.
"""

from typing import Any

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QTableWidget,
    QVBoxLayout,
    QWidget,
)

from .documents import IncomingPipeline, unique_name


class ExportDialog(QDialog):
    """
    Choose the current pipeline and named presets to include in one export.
    """

    def __init__(self, parent: QWidget, names: list[str], current_name: str) -> None:
        """
        Initially select only the current pipeline, including its open edits.
        """
        super().__init__(parent)
        self.setWindowTitle("Export pipelines")
        self.resize(560, 400)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Check the pipelines to include in the export."))
        self.listing = QListWidget()
        for name in names:
            label = (
                f"{name} — current pipeline (includes open edits)" if name == current_name else name
            )
            item = QListWidgetItem(label, self.listing)
            item.setData(Qt.ItemDataRole.UserRole, name)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(
                Qt.CheckState.Checked if name == current_name else Qt.CheckState.Unchecked
            )
        layout.addWidget(self.listing)
        selection = QHBoxLayout()
        for title, checked in (("Select all", True), ("Clear selection", False)):
            button = QPushButton(title)
            button.clicked.connect(lambda _=False, value=checked: self.check_all(value))
            selection.addWidget(button)
        layout.addLayout(selection)
        self.buttons = QDialogButtonBox()
        self.buttons.addButton(QDialogButtonBox.StandardButton.Ok)
        self.buttons.addButton(QDialogButtonBox.StandardButton.Cancel)
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Export selected")
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        self.listing.itemChanged.connect(self.update_selection)
        layout.addWidget(self.buttons)
        self.update_selection()

    def check_all(self, checked: bool) -> None:
        """
        Select or clear the entire export checklist.
        """
        for row in range(self.listing.count()):
            self.listing.item(row).setCheckState(
                Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked
            )

    def selected_names(self) -> list[str]:
        """
        Return checked names in their visible order.
        """
        return [
            self.listing.item(row).data(Qt.ItemDataRole.UserRole)
            for row in range(self.listing.count())
            if self.listing.item(row).checkState() == Qt.CheckState.Checked
        ]

    def update_selection(self) -> None:
        """
        Prevent an empty export.
        """
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setEnabled(
            bool(self.selected_names())
        )


class ImportDialog(QDialog):
    """
    Choose incoming pipelines and resolve every name before importing anything.
    """

    def __init__(
        self, parent: QWidget, incoming: list[IncomingPipeline], existing: set[str]
    ) -> None:
        """
        Suggest unique names for collisions, keeping existing presets by default.
        """
        super().__init__(parent)
        self.setWindowTitle("Import pipelines")
        self.resize(800, 440)
        self.incoming, self.existing = incoming, existing
        self.rows: list[tuple[QCheckBox, QLineEdit, QComboBox]] = []
        layout = QVBoxLayout(self)
        heading = QLabel(
            "Check pipelines to import. Resolve name conflicts with Rename, Replace or Skip."
        )
        heading.setWordWrap(True)
        layout.addWidget(heading)
        self.table = QTableWidget(len(incoming), 4)
        self.table.setHorizontalHeaderLabels(["Include", "Pipeline / file", "Import as", "Action"])
        occupied = set(existing)
        for row, item in enumerate(incoming):
            included = QCheckBox()
            included.setChecked(True)
            included.setAccessibleName(f"Include {item.name}")
            name = QLineEdit(unique_name(item.name, occupied))
            occupied.add(name.text())
            name.setAccessibleName(f"Import name for {item.name}")
            action = QComboBox()
            action.addItems(["Rename", "Replace", "Skip"])
            action.setAccessibleName(f"Conflict action for {item.name}")
            self.table.setCellWidget(row, 0, included)
            label = QLabel(f"{item.name}\n{item.source}")
            label.setToolTip(label.text())
            self.table.setCellWidget(row, 1, label)
            self.table.setCellWidget(row, 2, name)
            self.table.setCellWidget(row, 3, action)
            self.rows.append((included, name, action))
            included.toggled.connect(self.update_selection)
            name.textChanged.connect(self.update_selection)
            action.currentTextChanged.connect(lambda _, index=row: self.change_action(index))
        self.table.resizeRowsToContents()
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        layout.addWidget(self.table)
        selection = QHBoxLayout()
        for title, checked in (("Select all", True), ("Clear selection", False)):
            button = QPushButton(title)
            button.clicked.connect(lambda _=False, value=checked: self.check_all(value))
            selection.addWidget(button)
        layout.addLayout(selection)
        self.load_first = QCheckBox("Load first selected pipeline after import")
        self.load_first.setChecked(True)
        layout.addWidget(self.load_first)
        self.status = QLabel()
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.buttons = QDialogButtonBox()
        self.buttons.addButton(QDialogButtonBox.StandardButton.Ok)
        self.buttons.addButton(QDialogButtonBox.StandardButton.Cancel)
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Import selected")
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)
        self.update_selection()

    def check_all(self, checked: bool) -> None:
        """
        Select or clear all incoming pipelines.
        """
        for included, _, _ in self.rows:
            included.setChecked(checked)

    def change_action(self, row: int) -> None:
        """
        Restore the original name for Replace or suggest a unique name for Rename.
        """
        _, name, action = self.rows[row]
        name.setEnabled(action.currentText() == "Rename")
        if action.currentText() == "Replace":
            name.setText(self.incoming[row].name)
        elif action.currentText() == "Rename":
            occupied = self.existing | {
                field.text().strip()
                for index, (_, field, _) in enumerate(self.rows)
                if index != row
            }
            name.setText(unique_name(self.incoming[row].name, occupied))
        self.update_selection()

    def selected_pipelines(self) -> dict[str, dict[str, Any]]:
        """
        Validate the complete selection, rejecting unresolved and duplicate names.
        """
        selected = {}
        for item, (included, field, action) in zip(self.incoming, self.rows, strict=True):
            if not included.isChecked() or action.currentText() == "Skip":
                continue
            name = field.text().strip()
            if not name:
                raise ValueError("Enter a name for every included pipeline.")
            if name in selected:
                raise ValueError(
                    f'Two selected pipelines use the name "{name}". Rename or skip one.'
                )
            if name in self.existing and action.currentText() != "Replace":
                raise ValueError(f'"{name}" already exists. Choose a new name or Replace.')
            selected[name] = item.document
        return selected

    def update_selection(self) -> None:
        """
        Explain unresolved conflicts and allow import only when the selection is valid.
        """
        if not hasattr(self, "buttons"):
            return
        try:
            selected = self.selected_pipelines()
            self.status.setText(
                f"{len(selected)} pipeline(s) selected."
                if selected
                else "Select at least one pipeline."
            )
            valid = bool(selected)
        except ValueError as exc:
            self.status.setText(str(exc))
            valid = False
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setEnabled(valid)
