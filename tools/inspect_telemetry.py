"""Inspect saved Duo USB frames without opening or changing the camera.

Usage: .venv/bin/python tools/inspect_telemetry.py diagnostics/full-frame-*.bin
Offsets in this report are bytes from the start of the assembled frame.
Candidate interpretations are observations, not a validated protocol schema.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import zlib
from pathlib import Path

import numpy as np

from topdon_duo.camera import HEADER_U16, decode_duo_frame

HEADER_BYTES = HEADER_U16 * 2


def inspect_frame(path: Path) -> tuple[dict, np.ndarray]:
    data = path.read_bytes()
    header, raw, _preview = decode_duo_frame(data)
    header_bytes = data[:HEADER_BYTES]

    def u32(offset: int) -> int:
        return struct.unpack_from("<I", header_bytes, offset)[0]

    def floats(offset: int, count: int) -> list[float]:
        return list(struct.unpack_from(f"<{count}f", header_bytes, offset))

    records = [header_bytes[208 + i * 208 : 416 + i * 208] for i in range(21)]
    report = {
        "file": str(path),
        "frame_bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
        "header_nonzero_u16_count": int(np.count_nonzero(header)),
        "observed": {
            "magic_at_0": f"0x{u32(0):08x}",
            "length_candidate_at_4": u32(4),
            "counter_candidates_at_40_84": [u32(40), u32(84)],
            "date_time_candidates_u16_at_48_104": [
                list(struct.unpack_from("<8H", header_bytes, offset)) for offset in (48, 104)
            ],
            "dimensions_candidates_at_64_68_88_92": [u32(offset) for offset in (64, 68, 88, 92)],
            "rate_candidate_at_76": u32(76),
            "plane_size_candidate_at_96": u32(96),
            "float32_at_128_132_136_140": floats(128, 4),
            "externally_claimed_scene_stats_float32_at_144": floats(144, 3),
            "repeated_208_byte_records": {
                "start": 208,
                "count": len(records),
                "u32_at_record_plus_8": [
                    struct.unpack_from("<I", record, 8)[0] for record in records
                ],
                "differing_byte_offsets_against_first_record": [
                    [i for i, (a, b) in enumerate(zip(records[0], record)) if a != b]
                    for record in records
                ],
                "first_record_float32_at_plus_12_36_80": [
                    floats(208 + offset, 1)[0] for offset in (12, 36, 80)
                ],
            },
            "changing_trailer_word_at_4632": f"0x{u32(4632):08x}",
            "end_marker_at_4636": f"0x{u32(4636):08x}",
            "trailer_matches_standard_crc32_header_prefix": (
                u32(4632) == zlib.crc32(header_bytes[:4632])
            ),
        },
        "thermal_plane_raw_stats": {
            "minimum": int(raw.min()),
            "average": float(raw.mean()),
            "maximum": int(raw.max()),
        },
        "aligned_nonzero_u32_header": [
            {"byte_offset": offset, "value": u32(offset)}
            for offset in range(0, HEADER_BYTES, 4)
            if u32(offset) != 0
        ],
    }
    return report, header


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("frames", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, help="Save the JSON report to this file.")
    args = parser.parse_args()
    reports, headers = [], []
    for path in args.frames:
        try:
            report, header = inspect_frame(path)
        except (OSError, ValueError) as exc:
            parser.error(f"{path}: {exc}")
        reports.append(report)
        headers.append(header)
    matrix = np.stack(headers)
    changed = np.flatnonzero(np.any(matrix != matrix[0], axis=0))
    output = {
        "interpretation_status": "Exploratory; field meanings require validation.",
        "header_bytes": HEADER_BYTES,
        "samples": reports,
        "changing_u16_fields": [
            {"byte_offset": int(index * 2), "values": matrix[:, index].tolist()}
            for index in changed
        ],
    }
    encoded = json.dumps(output, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(encoded)
    else:
        print(encoded, end="")


if __name__ == "__main__":
    main()
