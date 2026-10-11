"""
Bounded JSON messages and binary payloads; never deserialize executable objects.
"""

import json
import struct
from typing import BinaryIO

MAX_HEADER = 16_384
MAX_PAYLOAD = 512_000


def read_exact(stream: BinaryIO, size: int) -> bytes:
    """
    Read a complete field or reject a disconnected/truncated peer.
    """
    chunks = bytearray()
    while len(chunks) < size:
        chunk = stream.read(size - len(chunks))
        if not chunk:
            raise EOFError("USB helper disconnected")
        chunks.extend(chunk)
    return bytes(chunks)


def receive(stream: BinaryIO) -> tuple[dict, bytes]:
    """
    Validate lengths and metadata before allocating or dispatching a message.
    """
    header_size, payload_size = struct.unpack("!II", read_exact(stream, 8))
    if not 0 < header_size <= MAX_HEADER or payload_size > MAX_PAYLOAD:
        raise ValueError("Invalid USB helper message size")
    metadata = json.loads(read_exact(stream, header_size))
    if not isinstance(metadata, dict):
        raise TypeError("Invalid USB helper metadata")
    payload = read_exact(stream, payload_size)
    return metadata, payload


def send(stream: BinaryIO, metadata: dict, payload: bytes = b"") -> None:
    """
    Write one message; callers serialize concurrent producers with a lock.
    """
    header = json.dumps(metadata, allow_nan=False).encode("utf-8")
    if not 0 < len(header) <= MAX_HEADER or len(payload) > MAX_PAYLOAD:
        raise ValueError("Invalid USB helper message size")
    for field in (struct.pack("!II", len(header), len(payload)), header, payload):
        remaining = memoryview(field)
        while remaining:
            written = stream.write(remaining)
            if written is None or written <= 0:
                raise OSError("USB helper pipe did not accept data")
            remaining = remaining[written:]
    stream.flush()
