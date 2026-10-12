"""
Provide an explicit retry/diagnostic command for automatic startup runtime setup.
"""

import argparse
import logging
import os
from pathlib import Path

from . import ensure_runtime
from .probes import query

LOG = logging.getLogger(__name__)


def main() -> int:
    """
    Set up host inference without opening a camera, and report the selected runtime.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retry", action="store_true", help="Ignore the failed-download cooldown")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ensure_runtime(retry=args.retry)
    selected = os.environ.get("BASED_THERMALS_ORT_PATH")
    result = query("cpu", Path(selected) if selected else None)
    LOG.info("Selected runtime CPU inference: %s", result["reason"])
    return 0 if result.get("available") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
