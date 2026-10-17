#!/usr/bin/env python3
"""Model-only SHA-256 identity with a stat-validated local cache."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CACHE_PATH = REPO_ROOT / ".todo-orchestrator/runtime/model-identity-cache.json"
CACHE_SCHEMA = "v200-model-identity-cache/v1"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SIGNATURE_KEYS = {"resolved_path", "st_dev", "st_ino", "size", "mtime_ns", "ctime_ns"}


class ModelIdentityError(RuntimeError):
    """The model could not be assigned a stable file identity."""


class ModelChangedError(ModelIdentityError):
    """The model's file identity changed during identity calculation."""


def _stat_identity(path: str | os.PathLike[str]) -> dict[str, str | int]:
    resolved = Path(path).resolve(strict=True)
    stat = resolved.stat()
    return {
        "resolved_path": str(resolved),
        "st_dev": stat.st_dev,
        "st_ino": stat.st_ino,
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "ctime_ns": stat.st_ctime_ns,
    }


def _valid_signature(value: Any) -> bool:
    if not isinstance(value, dict) or set(value) != _SIGNATURE_KEYS:
        return False
    if not isinstance(value.get("resolved_path"), str):
        return False
    return all(type(value.get(key)) is int for key in _SIGNATURE_KEYS - {"resolved_path"})


def _read_cache(path: Path) -> dict[str, dict[str, Any]]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return {}
    if (not isinstance(value, dict) or value.get("schema") != CACHE_SCHEMA or
            not isinstance(value.get("models"), dict)):
        return {}
    entries: dict[str, dict[str, Any]] = {}
    for key, entry in value["models"].items():
        if (not isinstance(key, str) or not isinstance(entry, dict) or
                not _valid_signature(entry.get("signature")) or
                entry["signature"].get("resolved_path") != key or
                not isinstance(entry.get("sha256"), str) or
                _SHA256.fullmatch(entry["sha256"]) is None):
            continue
        entries[key] = {"signature": entry["signature"], "sha256": entry["sha256"]}
    return entries


def _write_cache(path: Path, entries: dict[str, dict[str, Any]]) -> None:
    temporary: str | None = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp",
                                         dir=path.parent)
        with os.fdopen(fd, "w", encoding="utf-8") as output:
            json.dump({"schema": CACHE_SCHEMA, "models": entries}, output,
                      sort_keys=True, separators=(",", ":"))
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
        temporary = None
    except OSError:
        # The cache only avoids hashing; a fresh, stable hash remains usable
        # when the local cache directory is unavailable.
        return
    finally:
        if temporary is not None:
            try:
                os.unlink(temporary)
            except OSError:
                pass


def _sha256_file(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def identify_model(path: str | os.PathLike[str], *,
                   cache_path: str | os.PathLike[str] = DEFAULT_CACHE_PATH
                   ) -> dict[str, Any]:
    """Return SHA-256, provenance, and the checked model file signature.

    Cache records are keyed by resolved path and reused only when every
    signature field matches. Callers remain responsible for comparing the
    returned digest to their pinned model SHA-256.
    """
    source_path = Path(path)
    cache_file = Path(cache_path)
    before = _stat_identity(source_path)
    resolved_path = Path(before["resolved_path"])
    cached = _read_cache(cache_file).get(str(resolved_path))
    if cached is not None and cached["signature"] == before:
        after = _stat_identity(source_path)
        if after != before:
            raise ModelChangedError(f"model changed during cached identity check: {source_path}")
        return {"sha256": cached["sha256"], "provenance": "cached_stat_match",
                "signature": before}

    digest = _sha256_file(resolved_path)
    after = _stat_identity(source_path)
    if after != before:
        raise ModelChangedError(f"model changed during SHA-256 hashing: {source_path}")

    entries = _read_cache(cache_file)
    entries[str(resolved_path)] = {"signature": before, "sha256": digest}
    _write_cache(cache_file, entries)
    return {"sha256": digest, "provenance": "fresh_hash", "signature": before}
