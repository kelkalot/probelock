"""Read and write probe.lock files (plain JSON, commit it to your repo)."""

from __future__ import annotations

import json
from pathlib import Path

from .models import Lockfile


def write_lockfile(lock: Lockfile, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(lock.to_dict(), indent=2) + "\n")


def read_lockfile(path: Path) -> Lockfile:
    return Lockfile.from_dict(json.loads(Path(path).read_text()))
