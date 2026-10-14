# /// script
# requires-python = ">=3.12"
# dependencies = ["onnx==1.20.1", "certifi>=2025.1"]
# ///
"""
Download pinned HFHPE weights and prepare a segmentation-free detector outside the viewer.
"""

import argparse
import hashlib
import importlib
import json
import ssl
import tarfile
from pathlib import Path
from urllib.request import urlopen

BASE_URL = (
    "https://github.com/PINTO0309/High-Angle_Robust_Fast_Head_Pose_Estimation/"
    "releases/download/weights/"
)
DETECTOR_FILE = "deimv2_dinov3_s_wholebody49_boxes_only.onnx"
DETECTOR_SHA = "9b64e68cd6cf2e085613ae1dd06a42941fc0b7fce57cd80441dfb45ce90a1323"
DETECTORS = {
    "dinov3-s": (DETECTOR_FILE, DETECTOR_SHA, "imagenet"),
    "hgnet-n": (
        "deimv2_hgnetv2_n_wholebody49_boxes_only.onnx",
        "155c3c6f37dcf398288735f96a18738b67eb8f7d02ec54f024c75bed0a169e31", "div255",
    ),
}
ARCHIVE_FILE = "vitt_distill_064_yp_unified_v10.tar.gz"
ARCHIVE_SHA = "ac06a442b9a0d9e308250ff7792a2cd3819c8597dfe3031c931bacd0b1379f39"
POSE_MEMBER = "vitt_distill_064_yp_unified_v10/hfhpe_vitt_rollgate_1x3x64x64.onnx"
POSE_SHA = "be054658797c1610f8ac82a2ec0a4a5b5aff6df7c9efb459120133b1fd4a6895"


def checksum(path: Path) -> str:
    """
    Hash large external weights without retaining a second copy in memory.
    """
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return digest


def download(path: Path, filename: str, expected: str, offline: bool) -> None:
    """
    Reuse checksum-verified files and reject changed release assets.
    """
    if path.is_file() and checksum(path) == expected:
        return
    if offline:
        raise ValueError(f"Missing or changed external asset: {path}")
    temporary = path.with_suffix(path.suffix + ".partial")
    try:
        certifi = importlib.import_module("certifi")
        context = ssl.create_default_context(cafile=certifi.where())
        with (urlopen(BASE_URL + filename, timeout=60, context=context) as response,
              temporary.open("wb") as target):
            total = 0
            while block := response.read(1024 * 1024):
                total += len(block)
                if total > 200_000_000:
                    raise ValueError("Head Pose download exceeds its size limit")
                target.write(block)
        if checksum(temporary) != expected:
            raise ValueError(f"Upstream checksum changed: {filename}; review before updating pins")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def prepare_detector(
    source: Path, destination: Path, source_sha: str, normalization: str,
) -> dict[str, int | str]:
    """
    Keep only the detection output's dependencies and mark its verified input contract.

    Never execute the original graph's unused segmentation branch. Extractor performs
    structured ONNX dependency traversal rather than rewriting tensor names by text.
    """
    onnx = importlib.import_module("onnx")
    extractor = importlib.import_module("onnx.utils").Extractor
    original = onnx.load(source)
    extracted = extractor(original).extract_model(["images"], ["label_xyxy_score"])
    properties = {value.key: value.value for value in original.metadata_props}
    properties.update({"based_thermals.detector_pruned": "1",
                       "based_thermals.detector_norm": normalization,
                       "based_thermals.source_sha256": source_sha})
    onnx.helper.set_model_props(extracted, properties)
    onnx.checker.check_model(extracted)
    temporary = destination.with_suffix(".partial.onnx")
    try:
        onnx.save(extracted, temporary)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    report = {"original_nodes": len(original.graph.node),
              "prepared_nodes": len(extracted.graph.node), "sha256": checksum(destination)}
    return report


def main() -> None:
    """
    Prepare local weights with provenance; never modify runtime preferences or dependencies.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("diagnostics/head-pose/models"))
    parser.add_argument("--offline", action="store_true", help="Require existing pinned downloads")
    parser.add_argument("--detector", choices=tuple(DETECTORS), default="hgnet-n")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    detector = args.output / "detector-original.onnx"
    archive = args.output / "pose.tar.gz"
    detector_file, detector_sha, normalization = DETECTORS[args.detector]
    download(detector, detector_file, detector_sha, args.offline)
    download(archive, ARCHIVE_FILE, ARCHIVE_SHA, args.offline)
    pose_path = args.output / "pose.onnx"
    with tarfile.open(archive, "r:gz") as bundle:
        member = bundle.getmember(POSE_MEMBER)
        if not member.isfile() or member.size > 70_000_000:
            raise ValueError("Pose archive member must be a bounded regular file")
        stream = bundle.extractfile(member)
        if stream is None:
            raise ValueError("Pose archive member is missing")
        with stream:
            pose = stream.read()
    if hashlib.sha256(pose).hexdigest() != POSE_SHA:
        raise ValueError("Pose model checksum changed")
    pose_path.write_bytes(pose)
    report = prepare_detector(detector, args.output / "detector.onnx", detector_sha, normalization)
    report.update(detector_source=BASE_URL + detector_file, detector_source_sha256=detector_sha,
                  pose_source=BASE_URL + ARCHIVE_FILE, pose_archive_sha256=ARCHIVE_SHA,
                  pose_member=POSE_MEMBER, pose_sha256=POSE_SHA)
    (args.output / "provenance.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
