import ctypes as C
from collections import deque
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import usb.core
from usb.backend import libusb1

from topdon_duo.queued_usb import QueuedBulkReader


class FakeLibrary:
    def __init__(self):
        self.pending = {}
        self.deliveries = deque()
        self.freed = []
        self.libusb_alloc_transfer = Mock(
            side_effect=lambda _count: C.pointer(libusb1._libusb_transfer())
        )
        self.libusb_submit_transfer = Mock(side_effect=self.submit)
        self.libusb_cancel_transfer = Mock(side_effect=self.cancel)
        self.libusb_handle_events_timeout_completed = Mock(side_effect=self.events)
        self.libusb_free_transfer = Mock(side_effect=self.free)

    def submit(self, transfer):
        self.pending[C.addressof(transfer.contents)] = transfer
        return 0

    def cancel(self, transfer):
        self.deliveries.append((C.addressof(transfer.contents), b"", 3))
        return 0

    def events(self, *_args):
        if self.deliveries:
            address, packet, status = self.deliveries.popleft()
            if address is None:
                address = next(iter(self.pending))
            transfer = self.pending.pop(address)
            native = transfer.contents
            C.memmove(native.buffer, packet, len(packet))
            native.actual_length = len(packet)
            native.status = status
            native.callback(transfer)
        return 0

    def free(self, transfer):
        address = C.addressof(transfer.contents)
        assert address not in self.pending
        self.freed.append(address)

    def device(self):
        return SimpleNamespace(_ctx=SimpleNamespace(
            backend=SimpleNamespace(lib=self, ctx=C.c_void_p()),
            managed_open=lambda: SimpleNamespace(handle=C.c_void_p(1)),
        ))


def test_requests_stay_queued_and_packet_bytes_are_owned_by_consumer():
    lib = FakeLibrary()
    with QueuedBulkReader(lib.device(), 0x81, 16384, 2000, 4) as reader:
        assert len(lib.pending) == 4
        lib.deliveries.append((None, b"first", 0))
        assert reader.read() == b"first"
        assert len(lib.pending) == 4
        lib.deliveries.append((None, b"second", 0))
        assert reader.read() == b"second"
        assert reader.statuses[0] == 2
    assert len(lib.freed) == 4
    assert not lib.pending


def test_partial_timeout_data_is_preserved_and_status_is_counted():
    lib = FakeLibrary()
    with QueuedBulkReader(lib.device(), 0x81, 16384, 2000, 2) as reader:
        lib.deliveries.append((None, b"partial", 2))
        assert reader.read() == b"partial"
        assert reader.statuses[2] == 1
        assert len(lib.pending) == 2


def test_native_errors_reach_reader_and_all_transfers_are_cancelled():
    lib = FakeLibrary()
    with QueuedBulkReader(lib.device(), 0x81, 16384, 2000, 4) as reader:
        lib.deliveries.append((None, b"", 5))
        with pytest.raises(usb.core.USBError, match="status 5"):
            reader.read()
    assert len(lib.freed) == 4
    assert not lib.pending


def test_partial_submission_failure_frees_only_after_cancellation():
    lib = FakeLibrary()
    original = lib.submit
    count = [0]

    def submit(transfer):
        count[0] += 1
        return original(transfer) if count[0] <= 2 else -1

    lib.libusb_submit_transfer.side_effect = submit
    with pytest.raises(usb.core.USBError), QueuedBulkReader(lib.device(), 0x81, 16384, 2000, 4):
        pass
    assert len(lib.freed) == 4
    assert not lib.pending


def test_slow_consumer_fails_instead_of_silently_dropping_packets():
    lib = FakeLibrary()
    with QueuedBulkReader(lib.device(), 0x81, 16384, 2000, 2) as reader:
        reader._packets.extend([b"packet"] * 512)
        lib.deliveries.append((None, b"overflow", 0))
        reader._events()
        with pytest.raises(usb.core.USBError, match="fell behind"):
            reader.read()
