"""Provenance capture.

A Phase-0 verdict is only worth anything if someone can later ask "which weights,
which frames, which code, which library versions produced this?" and get an exact
answer. Everything here exists to make that answerable from the artifacts alone.
"""

from __future__ import annotations

import hashlib
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def sha256_file(path: Path | str, chunk_size: int = 1 << 20) -> str:
    """Streaming SHA-256 of a file's bytes."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def file_facts(path: Path | str) -> dict:
    """Identity of a file: absolute path, digest, size, mtime."""
    path = Path(path).resolve()
    stat = path.stat()
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "size_bytes": stat.st_size,
        "mtime_utc": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
    }


def environment_facts(device: str | None = None) -> dict:
    """Interpreter, platform, and the library versions that shape inference."""
    versions: dict[str, str] = {}
    for module_name, key in (
        ("ultralytics", "ultralytics"),
        ("torch", "torch"),
        ("cv2", "opencv"),
        ("numpy", "numpy"),
    ):
        try:
            module = __import__(module_name)
            versions[key] = str(getattr(module, "__version__", "unknown"))
        except Exception:  # noqa: BLE001 - a missing optional library is just absent
            versions[key] = "not-installed"

    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "device": device,
        "libraries": versions,
    }


def git_facts(repo_dir: Path | str) -> dict:
    """Best-effort commit identity. A missing or broken git is not an error."""
    repo_dir = Path(repo_dir)

    def run(*args: str) -> str | None:
        try:
            result = subprocess.run(
                ["git", *args],
                cwd=repo_dir,
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return result.stdout.strip() if result.returncode == 0 else None

    commit = run("rev-parse", "HEAD")
    if commit is None:
        return {"available": False, "commit": None, "dirty": None}
    status = run("status", "--porcelain")
    return {
        "available": True,
        "commit": commit,
        "dirty": bool(status) if status is not None else None,
    }


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def output_inventory(root: Path | str, paths: list[Path]) -> list[dict]:
    """Digest every artifact written, relative to the run directory.

    Lets a reviewer prove the records they read are the records the tool wrote.
    """
    root = Path(root)
    inventory = []
    for path in sorted(paths):
        if not path.is_file():
            continue
        inventory.append(
            {
                "path": str(path.relative_to(root)),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        )
    return inventory
