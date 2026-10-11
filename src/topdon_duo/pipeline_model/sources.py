"""
Resolve connected thermal/optical source usage without decoding frames or opening devices.
"""

from typing import Any


def source_required(document: dict[str, Any], source: str) -> bool:
    """
    Include connected branch sources, direct blend inputs and masks, ignoring bypasses.
    """
    from .document import execution_dependencies, software_tabs, thermal_source

    tabs = software_tabs(document)
    for tab in execution_dependencies(document):
        nodes = tabs[tab]
        selected = nodes[0]["params"]["source"]
        if selected == source and not (source == "preview" and thermal_source({"software": nodes})):
            return True
        if any(
            node["type"] == "combine"
            and not node["bypass"]
            and source in (node["params"]["tab"], node["params"]["mask_source"])
            for node in nodes
        ):
            return True
    return False


def display_source(document: dict[str, Any]) -> str:
    """
    Preserve the selected optical source and thermal range fallback in status metadata.
    """
    from .document import thermal_source

    return "raw" if thermal_source(document) else document["software"][0]["params"]["source"]
