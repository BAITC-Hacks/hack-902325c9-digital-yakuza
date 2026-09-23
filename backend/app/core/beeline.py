import atexit
import hashlib
import json
import sys
import tempfile
from functools import lru_cache
from pathlib import Path
from zipfile import ZipFile

RUN_TIMEOUT_SECONDS = 600


@lru_cache
def agent_directory() -> Path:
    archive_path = Path(__file__).resolve().parents[2] / "vendor" / "beeline_agent.zip"
    temporary = tempfile.TemporaryDirectory(prefix="beeline-agent-")
    atexit.register(temporary.cleanup)
    directory = Path(temporary.name)
    with ZipFile(archive_path) as archive:
        for member in archive.infolist():
            target = (directory / member.filename).resolve()
            if not target.is_relative_to(directory):
                raise ValueError("Invalid agent archive path")
        archive.extractall(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    for name, digest in manifest.items():
        if hashlib.sha256((directory / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Agent package checksum mismatch: {name}")
    sys.path.insert(0, str(directory))
    return directory
