"""
Stable pipeline tab identities and safe branch creation/removal.
"""

from typing import Any


def is_tab_name(value: Any) -> bool:
    """
    Recognize uppercase ASCII tab identities, including labels after Z.
    """
    return isinstance(value, str) and 1 <= len(value) <= 80 and all("A" <= c <= "Z" for c in value)


def tab_names(document: dict[str, Any]) -> list[str]:
    """
    Return A followed by stable alphabetic branch labels in spreadsheet order.
    """
    return ["A", *sorted(document["branches"], key=lambda label: (len(label), label))]


def add_pipeline_tab(document: dict[str, Any]) -> str:
    """
    Add the first unused label with its pinned image source and return that label.
    """
    from .document import node

    used = set(tab_names(document))
    index = 2
    while True:
        number, label = index, ""
        while number:
            number, remainder = divmod(number - 1, 26)
            label = chr(65 + remainder) + label
        if label not in used:
            document["branches"][label] = [node("software", "source")]
            return label
        index += 1


def tab_removal_reason(document: dict[str, Any], tab: str) -> str:
    """
    Explain pinned output or any saved incoming/outgoing blend or mask connection.
    """
    if tab == "A":
        return "Tab A contains the viewer output and cannot be removed."
    if tab not in document["branches"]:
        return "This pipeline tab no longer exists."
    tabs = {"A": document["software"], **document["branches"]}
    for owner, nodes in tabs.items():
        for item in nodes:
            if item["type"] != "combine":
                continue
            for key in ("tab", "mask_source"):
                target = item["params"][key]
                if target in tabs and tab in (owner, target):
                    return f"Tab {tab} is connected to another pipeline. Remove its blend/mask links first."
    return ""


def remove_pipeline_tab(document: dict[str, Any], tab: str) -> None:
    """
    Remove an unconnected auxiliary tab without renaming any surviving identities.
    """
    reason = tab_removal_reason(document, tab)
    if reason:
        raise ValueError(reason)
    del document["branches"][tab]
