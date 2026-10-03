import struct

import numpy as np
import pytest

from topdon_duo.camera import (
    FRAME_BYTES,
    FRAME_MAGIC,
    FRAME_U16,
    HEADER_U16,
    SENSOR_HEIGHT,
    SENSOR_PIXELS,
    SENSOR_WIDTH,
    FrameAssembler,
    build_probe,
    decode_duo_frame,
    parse_probe,
    raw_temperatures,
)
from topdon_duo.render import InvalidThermalFrame, ThermalRenderer


def make_frame(raw_value: int = 20_000) -> bytes:
    values = np.zeros(FRAME_U16, dtype="<u2")
    values[0] = FRAME_MAGIC & 0xFFFF
    values[1] = FRAME_MAGIC >> 16
    values[HEADER_U16 : HEADER_U16 + SENSOR_PIXELS] = raw_value
    return values.tobytes()


def test_probe_round_trip():
    probe = build_probe()
    struct.pack_into("<I", probe, 18, FRAME_BYTES)
    struct.pack_into("<I", probe, 22, 5020)
    mode = parse_probe(probe)
    assert mode.frame_index == 10
    assert mode.fps == 25
    assert mode.max_frame_size == FRAME_BYTES
    assert mode.max_payload_size == 5020


def test_frame_assembler_uses_fid_boundary():
    source = make_frame()
    assembler = FrameAssembler()
    result = None
    for offset in range(0, len(source), 5018):
        result = assembler.feed(bytes([2, 0]) + source[offset : offset + 5018]) or result
    result = assembler.feed(bytes([2, 1]) + b"next") or result
    assert result == source


def test_decode_and_temperature_conversion():
    frame = make_frame()
    telemetry, raw, preview = decode_duo_frame(frame)
    raw.reshape(-1)[:2_000] -= 64
    temperatures = raw_temperatures(raw, ambient_celsius=22.0)
    assert telemetry.shape == (HEADER_U16,)
    assert raw.shape == (SENSOR_HEIGHT, SENSOR_WIDTH)
    assert preview.shape == (SENSOR_HEIGHT, SENSOR_WIDTH)
    assert temperatures[0, 0] == pytest.approx(22.0)
    assert temperatures[100, 100] == pytest.approx(23.0)


def test_decode_rejects_wrong_magic():
    with pytest.raises(ValueError, match="invalid Duo frame magic"):
        decode_duo_frame(bytes(FRAME_BYTES))


def test_renderer_rejects_large_absurd_temperature_band():
    values = np.frombuffer(bytearray(make_frame()), dtype="<u2")
    temperature = values[HEADER_U16 : HEADER_U16 + SENSOR_PIXELS]
    temperature[: SENSOR_PIXELS // 5] = 50_000
    with pytest.raises(InvalidThermalFrame):
        ThermalRenderer().render(values.tobytes())
