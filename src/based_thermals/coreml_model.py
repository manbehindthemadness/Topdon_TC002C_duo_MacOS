"""Materialize ONNX's default convolution padding for the Core ML provider.

ORT 1.30's Core ML ConvTranspose checker dereferences an absent ``pads`` attribute,
and its Conv exporter can omit the Core ML-required ``pad`` parameter when ONNX
padding is implicit. Keep the bundled CPU models unchanged and add explicit zeros
only to the in-memory Apple copy. This small protobuf wire transformation needs
no ONNX developer dependency.
Unknown fields are preserved verbatim; only Model.graph / Graph.node are visited.
"""


from collections.abc import Callable, Iterator


def _varint(value: int) -> bytes:
    """
    Encode an unsigned protobuf integer.
    """
    result = bytearray()
    while value > 127:
        result.append((value & 127) | 128)
        value >>= 7
    result.append(value)
    return bytes(result)


def _read_varint(data: bytes, pos: int) -> tuple[int, int]:
    """
    Decode one protobuf integer and return its next byte offset.
    """
    value = 0
    for shift in range(0, 70, 7):
        if pos >= len(data):
            break
        byte = data[pos]
        pos += 1
        value |= (byte & 127) << shift
        if byte < 128:
            return value, pos
    raise ValueError("Invalid bundled ONNX protobuf varint")


def _fields(data: bytes | None) -> Iterator[tuple[int, bytes | None, bytes]]:
    """
    Read protobuf fields and reject missing or malformed payloads.
    """
    if data is None:
        raise ValueError("Missing ONNX protobuf field payload")
    pos = 0
    while pos < len(data):
        start = pos
        tag, pos = _read_varint(data, pos)
        field, wire = tag >> 3, tag & 7
        if not field:
            raise ValueError("Invalid bundled ONNX protobuf field")
        payload = None
        if wire == 2:
            size, pos = _read_varint(data, pos)
            payload = data[pos : pos + size]
            pos += size
        elif wire == 0:
            _, pos = _read_varint(data, pos)
        elif wire in (1, 5):
            pos += 8 if wire == 1 else 4
        else:
            raise ValueError("Unsupported bundled ONNX protobuf wire type")
        if pos > len(data):
            raise ValueError("Truncated bundled ONNX protobuf")
        yield field, payload, data[start:pos]


def _bytes_field(field: int, payload: bytes) -> bytes:
    """
    Encode one length-delimited protobuf field.
    """
    return _varint(field << 3 | 2) + _varint(len(payload)) + payload


def _rewrite(data: bytes, field: int, transform: Callable[[bytes], bytes]) -> bytes:
    """
    Transform selected protobuf fields while preserving other wire bytes.
    """
    return b"".join(
        _bytes_field(number, transform(payload))
        if number == field and payload is not None
        else original
        for number, payload, original in _fields(data)
    )


def explicit_coreml_padding(model: bytes) -> bytes:
    """
    Make default 2D convolution padding explicit without overriding auto_pad.
    """

    def patch_node(node: bytes) -> bytes:
        """
        Add default padding only when convolution metadata requires it.
        """
        fields = list(_fields(node))
        if not any(
            number == 4 and payload in (b"Conv", b"ConvTranspose") for number, payload, _ in fields
        ):
            return node
        for number, attr, _ in fields:
            if number == 5 and attr is not None:
                attributes = list(_fields(attr))
                if any(n == 1 and name == b"auto_pad" for n, name, _ in attributes) and any(
                    n == 4 and value not in (b"NOTSET", b"") for n, value, _ in attributes
                ):
                    return node
            if (
                number == 5
                and attr is not None
                and any(n == 1 and name == b"pads" for n, name, _ in _fields(attr))
            ):
                return node
        # AttributeProto: name=1, ints=8, type=20 (INTS=7).
        attribute = _bytes_field(1, b"pads") + b"\x40\x00" * 4 + _varint(20 << 3) + b"\x07"
        return node + _bytes_field(5, attribute)

    # ModelProto.graph=7; GraphProto.node=1; NodeProto.attribute=5.
    return _rewrite(model, 7, lambda graph: _rewrite(graph, 1, patch_node))


def coreml_input_shape_overrides(
    model: bytes,
    shape: tuple[int, ...],
    input_name: str = "x",
) -> dict[str, int]:
    """Specialize V1's symbolic NCHW dimensions without changing its weights.

    Core ML's MLProgram layout conversion fails with the V1 dynamic-shape export.
    ORT's free-dimension overrides let shape inference resolve the Apple graph.
    """
    graph = next(payload for n, payload, _ in _fields(model) if n == 7)
    for number, info, _ in _fields(graph):
        if number != 11 or not any(
            n == 1 and payload == input_name.encode() for n, payload, _ in _fields(info)
        ):
            continue
        value_type = next(p for n, p, _ in _fields(info) if n == 2)
        tensor_type = next(p for n, p, _ in _fields(value_type) if n == 1)
        tensor_shape = next(p for n, p, _ in _fields(tensor_type) if n == 2)
        dimensions = [p for n, p, _ in _fields(tensor_shape) if n == 1]
        if len(dimensions) != len(shape):
            raise ValueError("Unexpected MewZoom V1 input rank")
        overrides = {}
        for dim, value in zip(dimensions, shape, strict=True):
            symbol = next((p for n, p, _ in _fields(dim) if n == 2), None)
            if symbol is not None:
                name = symbol.decode()
                if name in overrides and overrides[name] != value:
                    raise ValueError("Conflicting MewZoom input dimensions")
                overrides[name] = value
        return overrides
    raise ValueError("MewZoom V1 input metadata missing")
