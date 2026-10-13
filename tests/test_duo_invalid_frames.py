"""
Reject damaged Duo frames before rendering and recover or stop cleanly at stream boundaries.
"""

import struct
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from support.frames import make_frame

from based_thermals.camera_backends.duo import camera as duo


@pytest.mark.parametrize("linux", [False, True])
@pytest.mark.parametrize("full_preview", [False, True])
@pytest.mark.parametrize("end_of_frame", [False, True])
def test_invalid_magic_is_rejected_and_next_frame_recovers(
    linux: bool, full_preview: bool, end_of_frame: bool,
) -> None:
    """
    Reject the reported payload prefix at EOF or FID boundaries for either supported layout.
    """
    source = make_frame()
    if full_preview:
        source += bytes(duo.YUY2_FRAME_BYTES - duo.FRAME_BYTES)
    damaged = struct.pack("<I", 0x801C801C) + source[4:]
    assembler = duo.LinuxFrameAssembler(len(source)) if linux else duo.FrameAssembler()
    if end_of_frame:
        assert assembler.feed(b"\x02\x82" + damaged) is None
        recovered = assembler.feed(b"\x02\x83" + source)
    else:
        assert assembler.feed(b"\x02\x80" + damaged) is None
        assert assembler.feed(b"\x02\x81" + source) is None
        recovered = assembler.feed(b"\x02\x83")
    assert recovered == source
    assert assembler.rejected["magic_mismatch"] == 1
    assert assembler.last_rejected_size == len(source)
    assert assembler.last_rejected_prefix == damaged[:16].hex()
    assert assembler.buffered_bytes == 0
    assert recovered is not None
    telemetry, counts, preview = duo.decode_duo_frame(recovered)
    assert telemetry.size == duo.HEADER_U16
    assert counts.shape == (duo.SENSOR_HEIGHT, duo.SENSOR_WIDTH)
    assert preview.shape == ((384, 512) if full_preview else (192, 256))


@pytest.mark.parametrize("platform", ["darwin", "linux"])
@pytest.mark.parametrize("queue_depth", [0, 32])
def test_stream_discards_invalid_frames_including_final_shutdown_payload(
    monkeypatch: pytest.MonkeyPatch, platform: str, queue_depth: int,
) -> None:
    """
    Keep valid frames flowing and terminate normally when stopping delivers damaged trailing bytes.
    """
    monkeypatch.setattr(duo.sys, "platform", platform)
    camera = duo.TC002CDuoCamera()
    camera.usb_queue_depth = queue_depth
    source = make_frame()
    damaged = struct.pack("<I", 0x801C801C) + source[4:]
    packets = iter((b"\x02\x82" + source, b"\x02\x83" + damaged,
                    b"\x02\x82" + source, b"\x02\x83" + damaged))
    reads = 0

    def read(_endpoint: int, _size: int, *, timeout: int) -> bytes:
        """
        Deliver a final damaged payload during a bounded stop, without touching USB.
        """
        nonlocal reads
        assert timeout == camera.timeout_ms
        packet = next(packets)
        reads += 1
        if reads == 4:
            camera.stop_stream()
        return packet

    def queued_reader(
        _device: object, endpoint: int, size: int, timeout: int, depth: int,
    ) -> nullcontext[SimpleNamespace]:
        """
        Exercise queued capture with the same fake payloads and no libusb transfers.
        """
        assert depth == queue_depth
        reader = SimpleNamespace(read=lambda: read(endpoint, size, timeout=timeout))
        return nullcontext(reader)

    # noinspection PyUnresolvedReferences
    monkeypatch.setattr(duo, "QueuedBulkReader", queued_reader)
    camera.device = Mock(read=read)
    camera.mode = duo.NegotiatedMode(1, 10, 400_000, duo.FRAME_BYTES, 5020)
    camera._running.set()
    assert list(camera.frames()) == [source, source]
    assert reads == 4
