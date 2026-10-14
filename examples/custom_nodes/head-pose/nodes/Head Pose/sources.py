"""
Pinned detector releases already verified to contain only detection dependencies.
"""

from typing import Any

DETECTORS = {
    "HGNet-N (fast)": {
        "name": "HFHPE DEIMv2 HGNet-N head detector",
        "url": "https://github.com/PINTO0309/High-Angle_Robust_Fast_Head_Pose_Estimation/releases/download/weights/deimv2_hgnetv2_n_wholebody49_boxes_only.onnx",
        "sha256": "155c3c6f37dcf398288735f96a18738b67eb8f7d02ec54f024c75bed0a169e31",
        "max_bytes": 20_000_000,
    },
    "DINOv3-S (reference)": {
        "name": "HFHPE DEIMv2 DINOv3-S head detector",
        "url": "https://github.com/PINTO0309/High-Angle_Robust_Fast_Head_Pose_Estimation/releases/download/weights/deimv2_dinov3_s_wholebody49_boxes_only.onnx",
        "sha256": "9b64e68cd6cf2e085613ae1dd06a42941fc0b7fce57cd80441dfb45ce90a1323",
        "max_bytes": 50_000_000,
    },
}

NORMALIZATIONS = {
    "155c3c6f37dcf398288735f96a18738b67eb8f7d02ec54f024c75bed0a169e31": "div255",
    "9b64e68cd6cf2e085613ae1dd06a42941fc0b7fce57cd80441dfb45ce90a1323": "imagenet",
}


def detector_source(choice: object) -> dict[str, Any]:
    """
    Resolve the saved UI choice without probing, importing models or downloading.
    """
    if not isinstance(choice, str) or choice not in DETECTORS:
        raise ValueError("Select HGNet-N (fast) or DINOv3-S (reference) as the head detector")
    source = dict(DETECTORS[choice])
    return source
