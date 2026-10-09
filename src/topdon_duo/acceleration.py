"""
Resolve portable GPU preferences without changing the saved pipeline.
"""


def select_backend(requested: str, apple_available: bool, nvidia_available: bool) -> str:
    """
    Prefer the requested GPU, then the other GPU, then CPU; honor explicit CPU.
    """
    if requested == "coreml":
        if apple_available:
            return "coreml"
        if nvidia_available:
            return "cuda"
    elif requested == "cuda":
        if nvidia_available:
            return "cuda"
        if apple_available:
            return "coreml"
    return "cpu"
