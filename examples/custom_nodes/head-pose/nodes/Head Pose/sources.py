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

POSES = {
    "ViT-T/16 128×128 (v11)": {
        "name": "HFHPE ViT-T/16 128 v11",
        "url": "https://github.com/PINTO0309/High-Angle_Robust_Fast_Head_Pose_Estimation/releases/download/weights/vitt_distill_128_yp_unified_v11.tar.gz",
        "archive_member": "vitt_distill_128_yp_unified_v11/hfhpe_vitt_1x3x128x128.onnx",
        "archive_sha256": "74a50a5c1cc6e85c2393bd61bf3d96f30a6049f398081f3d00b36b8640c17b31",
        "sha256": "02136d0746d0acad26c0bea6d405826d028c780d0ec6a81d43b93e33004946a7",
        "max_bytes": 200_000_000,
    },
    "ViT-T/16 128×128 (rollgate v11)": {
        "name": "HFHPE ViT-T/16 128 rollgate v11",
        "url": "https://github.com/PINTO0309/High-Angle_Robust_Fast_Head_Pose_Estimation/releases/download/weights/vitt_distill_128_yp_unified_v11.tar.gz",
        "archive_member": "vitt_distill_128_yp_unified_v11/hfhpe_vitt_rollgate_1x3x128x128.onnx",
        "archive_sha256": "74a50a5c1cc6e85c2393bd61bf3d96f30a6049f398081f3d00b36b8640c17b31",
        "sha256": "0f4311d7cf8560d2c04a2d40e7f166a8195ad50107d0a6b64336fc8fad517db8",
        "max_bytes": 200_000_000,
    },
}


def pose_source(choice: object, default: dict[str, Any]) -> dict[str, Any]:
    """
    Select an audited integrated pose export, retaining the original 64-pixel default.
    """
    if choice == "ViT-T/16 64×64 (rollgate v10)":
        return dict(default)
    if not isinstance(choice, str) or choice not in POSES:
        raise ValueError("Select a supported ViT-T/16 pose model")
    source = dict(POSES[choice])
    return source


def detector_source(choice: object) -> dict[str, Any]:
    """
    Resolve the saved UI choice without probing, importing models or downloading.
    """
    if not isinstance(choice, str) or choice not in DETECTORS:
        raise ValueError("Select HGNet-N (fast) or DINOv3-S (reference) as the head detector")
    source = dict(DETECTORS[choice])
    return source
