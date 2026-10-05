"""Validate saved measuring spots in unrotated, unmirrored sensor coordinates."""

from .camera import SENSOR_HEIGHT, SENSOR_WIDTH


def validate_spots(saved):
    if (
        not isinstance(saved, dict)
        or type(saved.get("version")) is not int
        or saved["version"] != 1
    ):
        raise ValueError("Unsupported saved spots")
    items = saved.get("items")
    if not isinstance(items, list) or len(items) > 1000:
        raise ValueError("Invalid saved spots")
    numbers = set()
    result = []
    for item in items:
        if not isinstance(item, dict):
            raise TypeError("Invalid saved spot")
        number, x, y = (item.get(key) for key in ("number", "x", "y"))
        name, enabled = item.get("name"), item.get("enabled")
        if (
            type(number) is not int
            or number < 1
            or number in numbers
            or type(x) is not int
            or not 0 <= x < SENSOR_WIDTH
            or type(y) is not int
            or not 0 <= y < SENSOR_HEIGHT
            or type(enabled) is not bool
            or not isinstance(name, str)
            or len(name) > 64
            or any(not char.isprintable() for char in name)
        ):
            raise TypeError("Invalid saved spot")
        numbers.add(number)
        result.append({"number": number, "x": x, "y": y, "name": name.strip(), "enabled": enabled})
    next_number, placing = saved.get("next_number"), saved.get("placing")
    if (
        type(next_number) is not int
        or next_number <= max(numbers, default=0)
        or type(placing) is not bool
    ):
        raise ValueError("Invalid saved spot numbering or placement state")
    return {"version": 1, "items": result, "next_number": next_number, "placing": placing}
