import struct

import numpy as np

from topdon_duo.camera import (
    FRAME_BYTES,
    FRAME_HEIGHT,
    FRAME_WIDTH,
    FrameAssembler,
    build_probe,
    decode_yuy2_frame,
    parse_probe,
    raw_temperatures,
)


def test_probe_round_trip():
    probe = build_probe()
    struct.pack_into("<I", probe, 18, FRAME_BYTES)
    struct.pack_into("<I", probe, 22, 5020)
    mode = parse_probe(probe)
    assert mode.frame_index == 1
    assert mode.fps == 25
    assert mode.max_frame_size == FRAME_BYTES
    assert mode.max_payload_size == 5020


def test_frame_assembler_uses_fid_boundary():
    source = bytes(index % 251 for index in range(FRAME_BYTES))
    assembler = FrameAssembler()
    result = None
    for offset in range(0, len(source), 5018):
        result = assembler.feed(bytes([2, 0]) + source[offset : offset + 5018]) or result
    result = assembler.feed(bytes([2, 1]) + b"next") or result
    assert result == source


def test_decode_and_temperature_conversion():
    array = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 2), dtype=np.uint8)
    kelvin_raw = int((25.0 + 273.15) * 64)
    array[FRAME_HEIGHT // 2 :, :, 0] = kelvin_raw & 0xFF
    array[FRAME_HEIGHT // 2 :, :, 1] = kelvin_raw >> 8
    image, radiometric = decode_yuy2_frame(array.tobytes())
    temperatures = raw_temperatures(radiometric)
    assert image.shape == (196, 256, 2)
    assert radiometric.shape == (196, 256, 2)
    assert np.allclose(temperatures, 25.0, atol=0.02)
