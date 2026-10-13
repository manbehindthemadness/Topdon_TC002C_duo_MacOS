"""
Render standalone measurement controls and independent device telemetry from metadata.
"""

from typing import Any

from .pipeline_ui.hardware import row_options


def update_hardware_ui(window: Any, state: dict[str, Any], locked: bool) -> None:
    """
    Add verified measurement controls while keeping unsupported saved controls visible disabled.
    """
    from .view_window import ControlRow

    capabilities = state.get("camera_capabilities")
    if not isinstance(capabilities, dict):
        return
    specs = capabilities.get("controls", {})
    for name, spec in specs.items():
        if (spec.get("effect") != "measurement"
                or spec.get("scope", "device") != "device" or not spec.get("supported")):
            continue
        description = {key: value for key, value in spec.items()
                       if key not in ("available", "reason", "supported")}
        if window.hardware_descriptions.get(name) == description:
            continue
        row = ControlRow(
            spec["title"], lambda value, field=name: window.send_hardware_command({
                "action": "hardware", "name": field, "value": value, "enabled": True,
            }), preserve_input=name == "ambient", **row_options(spec),
        )
        previous = window.hardware_rows.get(name)
        if previous is not None:
            previous.timer.stop()
            window.hardware_layout.replaceWidget(previous, row)
            previous.deleteLater()
        else:
            window.hardware_layout.insertWidget(window.hardware_layout.count() - 1, row)
        window.hardware_rows[name] = row
        window.hardware_descriptions[name] = description
    for name, row in window.hardware_rows.items():
        spec = specs.get(name, {})
        row.setVisible(bool(spec.get("supported")))
        row.setToolTip(spec.get("reason", ""))
    features = capabilities.get("features", {})
    window.auto_calibrate.setEnabled(not locked and bool(features.get("auto_calibrate")))
    for tool, name in ((window.distance_calibration, "distance"),
                       (window.emissivity_calibration, "emissivity"),
                       (window.reflected_calibration, "reflected")):
        tool.setVisible(bool(features.get(f"host_{name}_calibration")))
    labels = []
    for reading in state.get("reported_readings", []):
        if not isinstance(reading, dict):
            continue
        value = reading.get("value")
        shown = "—"
        if reading.get("valid") and isinstance(value, (int, float)) and not isinstance(value, bool):
            shown = f"{value:g}"
        spot = f" · Spot {reading['spot_id']}" if reading.get("spot_id") is not None else ""
        labels.append(f"{reading.get('name', 'Reading')}{spot}: {shown} {reading.get('unit', '')}")
    window.reported_readings.setText("\n".join(labels))
    window.reported_readings.setVisible(bool(labels))
