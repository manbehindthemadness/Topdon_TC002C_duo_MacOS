"""
Extract one bounded regular model file without unpacking an archive into the cache.
"""

import hashlib
import tarfile
import tempfile
from pathlib import Path, PurePosixPath


def extract_model(archive: Path, directory: Path, member: str, maximum: int) -> tuple[Path, str]:
    """
    Stream one regular tar.gz member to a temporary file and compute its checksum.
    """
    path = PurePosixPath(member)
    if path.is_absolute() or ".." in path.parts or path.as_posix() != member:
        raise ValueError("Invalid model archive member path")
    temporary: Path | None = None
    try:
        with tarfile.open(archive, mode="r|gz") as source:
            for index, entry in enumerate(source):
                if index >= 4096:
                    raise ValueError("Model archive contains too many entries")
                if entry.name != member:
                    continue
                if not entry.isfile() or not 0 < entry.size <= maximum:
                    raise ValueError("Model archive member is not a bounded regular file")
                stream = source.extractfile(entry)
                if stream is None:
                    raise ValueError("Model archive member is unreadable")
                digest, received = hashlib.sha256(), 0
                with stream, tempfile.NamedTemporaryFile(dir=directory, delete=False) as target:
                    temporary = Path(target.name)
                    while chunk := stream.read(1024 * 1024):
                        received += len(chunk)
                        if received > maximum:
                            raise ValueError("Extracted model exceeds the download size limit")
                        digest.update(chunk)
                        target.write(chunk)
                if received != entry.size:
                    raise ValueError("Model archive member is truncated")
                return Path(target.name), digest.hexdigest()
        raise ValueError(f"Model archive member not found: {member}")
    except (OSError, EOFError, ValueError, tarfile.TarError):
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise
