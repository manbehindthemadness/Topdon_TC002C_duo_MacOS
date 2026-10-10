"""Validate saved measuring spots in unrotated, unmirrored sensor coordinates."""

from .camera import SENSOR_HEIGHT, SENSOR_WIDTH


def validate_spots(saved: object, *, native_size: tuple[int, int] | None = None) -> dict:
    """
    Validate unrotated coordinates against the selected native sensor grid.
    """
    width, height = native_size or (SENSOR_WIDTH, SENSOR_HEIGHT)
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
            not isinstance(number, int) or isinstance(number, bool)
            or number < 1
            or number in numbers
            or not isinstance(x, int) or isinstance(x, bool)
            or not 0 <= x < width
            or not isinstance(y, int) or isinstance(y, bool)
            or not 0 <= y < height
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
        not isinstance(next_number, int) or isinstance(next_number, bool)
        or next_number <= max(numbers, default=0)
        or type(placing) is not bool
    ):
        raise ValueError("Invalid saved spot numbering or placement state")
    return {"version": 1, "items": result, "next_number": next_number, "placing": placing}
