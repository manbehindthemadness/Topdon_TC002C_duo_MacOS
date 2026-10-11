"""
Dynamic tab selection, branch controls and available combine sources.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtWidgets import QMessageBox

from ..pipeline import execution_dependencies
from ..pipeline_model.tabs import (
    add_pipeline_tab,
    remove_pipeline_tab,
    tab_names,
    tab_removal_reason,
)

if TYPE_CHECKING:
    from .editor import PipelineEditor


class PipelineTabs:
    """
    Manage editor tabs while preserving stable identities and the viewer output.
    """

    tab: str

    def sync_tabs(self: PipelineEditor) -> None:
        """
        Reconcile tab labels after edits, imports or an incoming document replacement.
        """
        names = tab_names(self.document)
        if self.tab not in names:
            self.tab = "A"
        self.tab_bar.blockSignals(True)
        while self.tab_bar.count():
            self.tab_bar.removeTab(0)
        for name in names:
            index = self.tab_bar.addTab(name)
            self.tab_bar.setTabData(index, name)
        self.tab_bar.setCurrentIndex(names.index(self.tab))
        self.tab_bar.blockSignals(False)

    def select_tab(self: PipelineEditor, index: int) -> None:
        """
        Commit debounced values before switching the displayed branch widgets.
        """
        if index < 0 or self.rebuilding:
            return
        if not self.locked:
            self.commit_pending_inputs()
        self.tab = self.tab_bar.tabData(index)
        self.rebuild()
        self.update_state(self.last_state, self.locked)

    def add_tab(self: PipelineEditor) -> None:
        """
        Create and select an empty branch through the normal pipeline edit boundary.
        """
        if self.locked:
            return
        self.commit_pending_inputs()
        self.tab = add_pipeline_tab(self.document)
        self.rebuild()
        self.publish()

    def remove_tab(self: PipelineEditor) -> None:
        """
        Remove only an unconnected branch and return selection to A.
        """
        if self.locked:
            return
        self.commit_pending_inputs()
        try:
            remove_pipeline_tab(self.document, self.tab)
        except ValueError as exc:
            QMessageBox.warning(self, "Cannot remove pipeline tab", str(exc))
            return
        self.tab = "A"
        self.rebuild()
        self.publish()

    def combine_options(self: PipelineEditor, key: str) -> tuple[tuple[str, str], ...]:
        """
        Offer native sources and existing branches other than the current branch.
        """
        sources = [("preview", "Camera preview"), ("raw", "Raw thermal")]
        if key == "mask_source":
            sources = [("none", "None"), ("input", "Blend input"), *sources]
        return tuple(sources + [(tab, f"Tab {tab}") for tab in tab_names(self.document) if tab != self.tab])

    def update_tab_status(self: PipelineEditor) -> None:
        """
        Present branch activity and explain why removal is unavailable.
        """
        try:
            active = execution_dependencies(self.document)
        except ValueError:
            active = {}
        self.tab_status.setText(
            "Tab A → viewer" if self.tab == "A"
            else f"Tab {self.tab} → connected" if self.tab in active
            else f"Tab {self.tab} · preview only" if any(
                n["type"] == "preview" and n["expanded"] and not n["bypass"]
                for n in self.nodes("software")
            )
            else f"Tab {self.tab} · idle — expand a Preview node or connect this tab to use its output"
        )
        for index, tab in enumerate(tab_names(self.document)):
            self.tab_bar.setTabToolTip(
                index, "Viewer output" if tab == "A" else "Connected" if tab in active else "Idle"
            )
        reason = tab_removal_reason(self.document, self.tab)
        self.add_tab_button.setEnabled(not self.locked)
        self.remove_tab_button.setEnabled(not self.locked and not reason)
        self.remove_tab_button.setToolTip(reason or f"Remove pipeline tab {self.tab}")
