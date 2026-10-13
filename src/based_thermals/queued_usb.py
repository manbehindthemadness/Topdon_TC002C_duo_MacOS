"""Queued bulk reads using the camera's existing PyUSB connection.

PyUSB exposes synchronous bulk reads publicly. Its libusb backend already
defines the native transfer ABI; keep the dependency on that private API here.
Each submitted transfer owns a separate buffer until cancellation completes.
"""

from __future__ import annotations

import ctypes as C
import threading
import time
from collections import Counter, deque

import usb.core
from usb.backend import libusb1


class _Timeval(C.Structure):
    _fields_ = [("seconds", C.c_long), ("microseconds", C.c_long)]


class QueuedBulkReader:
    def __init__(self, device, endpoint, size, timeout_ms, depth=32):
        if not 2 <= depth <= 128:
            raise ValueError("USB queue depth must be between 2 and 128")
        self._backend = device._ctx.backend
        self._lib = self._backend.lib
        self._handle = device._ctx.managed_open().handle
        self._endpoint, self._size = endpoint, size
        self._timeout_ms, self._depth = timeout_ms, depth
        self._transfers = []
        self._buffers = []
        self._pending = {}
        self._packets = deque()
        self._lock = threading.RLock()
        self._closing = False
        self._error = None
        self.statuses = Counter()
        self._callback = libusb1._libusb_transfer_cb_fn_p(self._completed)
        self._lib.libusb_cancel_transfer.argtypes = [libusb1._libusb_transfer_p]
        self._lib.libusb_cancel_transfer.restype = C.c_int
        self._lib.libusb_handle_events_timeout_completed.argtypes = [
            C.c_void_p, C.POINTER(_Timeval), C.POINTER(C.c_int)
        ]
        self._lib.libusb_handle_events_timeout_completed.restype = C.c_int

    def _submit(self, transfer):
        result = self._lib.libusb_submit_transfer(transfer)
        libusb1._check(result)
        self._pending[C.addressof(transfer.contents)] = transfer

    def __enter__(self):
        try:
            with self._lock:
                for _ in range(self._depth):
                    transfer = self._lib.libusb_alloc_transfer(0)
                    if not transfer:
                        raise MemoryError("Cannot allocate USB transfer")
                    self._transfers.append(transfer)
                    buffer = C.create_string_buffer(self._size)
                    self._buffers.append(buffer)
                    native = transfer.contents
                    native.dev_handle = self._handle
                    native.flags = 0
                    native.endpoint = self._endpoint
                    native.type = 2  # LIBUSB_TRANSFER_TYPE_BULK
                    native.timeout = self._timeout_ms
                    native.length = self._size
                    native.callback = self._callback
                    native.buffer = C.cast(buffer, C.c_void_p)
                    native.num_iso_packets = 0
                for transfer in self._transfers:
                    self._submit(transfer)
            return self
        except Exception:
            self.close()
            raise

    def _completed(self, transfer):
        # libusb may invoke this on a thread doing a control transfer, too.
        # Never let an exception escape a ctypes callback.
        with self._lock:
            native = transfer.contents
            self._pending.pop(C.addressof(native), None)
            self.statuses[native.status] += 1
            if self._closing:
                return
            try:
                if native.status not in (0, 2):  # completed or timed out
                    raise usb.core.USBError(f"Queued USB transfer status {native.status}")
                packet = C.string_at(native.buffer, native.actual_length)
                if len(self._packets) >= 512:
                    raise usb.core.USBError("Queued USB consumer fell behind (512 packets)")
                # Re-arm before parsing the payload; other requests remain queued.
                self._submit(transfer)
                if packet:
                    self._packets.append(packet)
            except Exception as exc:  # noqa: BLE001 - propagate on the reader thread
                self._error = exc

    def _events(self):
        timeout = _Timeval(0, 20_000)
        result = self._lib.libusb_handle_events_timeout_completed(
            self._backend.ctx, C.byref(timeout), None
        )
        if result != -10:  # LIBUSB_ERROR_INTERRUPTED is retryable.
            libusb1._check(result)

    def read(self):
        deadline = time.monotonic() + self._timeout_ms / 1000
        while True:
            with self._lock:
                if self._error is not None:
                    raise self._error
                if self._packets:
                    return self._packets.popleft()
                if self._closing:
                    raise usb.core.USBError("Queued USB reader is closed")
            if time.monotonic() >= deadline:
                raise usb.core.USBTimeoutError("Queued USB read timed out")
            self._events()

    def close(self):
        with self._lock:
            self._closing = True
            pending = list(self._pending.values())
        for transfer in pending:
            result = self._lib.libusb_cancel_transfer(transfer)
            if result not in (0, -5):  # NOT_FOUND can mean completion is already pending.
                libusb1._check(result)
        # A timeout is not permission to free buffers still owned by libusb.
        while self._pending:
            self._events()
        for transfer in self._transfers:
            self._lib.libusb_free_transfer(transfer)
        self._transfers.clear()
        self._buffers.clear()
        self._packets.clear()

    def __exit__(self, *_exc):
        self.close()
