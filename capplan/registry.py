"""Versioned artefact directory with a manifest.

Deliberately not MLflow. A run is a directory; the manifest is one JSON file
that records what went in, what came out, and the hash of every artefact. That
is enough to answer "which run produced the number in the board pack" a year
later, which is the only question the registry has to answer.

Point MLflow at these directories later if someone wants the UI.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

MANIFEST_NAME = "manifest.json"
_LATEST = "latest"


def _utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _git_rev() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):  # pragma: no cover - no git
        return None
    return out.stdout.strip() or None if out.returncode == 0 else None


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass
class Run:
    """A single versioned artefact directory."""

    run_id: str
    root: Path
    kind: str
    manifest: dict[str, Any] = field(default_factory=dict)

    @property
    def dir(self) -> Path:
        return self.root / self.kind / self.run_id

    def path(self, *parts: str) -> Path:
        target = self.dir.joinpath(*parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        return target

    # -- manifest -----------------------------------------------------

    def record(self, key: str, value: Any) -> None:
        self.manifest.setdefault("record", {})[key] = value

    def record_metrics(self, metrics: dict[str, Any]) -> None:
        self.manifest.setdefault("metrics", {}).update(metrics)

    def record_inputs(self, inputs: dict[str, Any]) -> None:
        self.manifest.setdefault("inputs", {}).update(inputs)

    def add_artefact(self, path: Path, role: str) -> None:
        rel = path.relative_to(self.dir) if path.is_absolute() else path
        entry = {
            "role": role,
            "path": str(rel),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        self.manifest.setdefault("artefacts", []).append(entry)

    def write_json(self, name: str, payload: Any, role: str | None = None) -> Path:
        target = self.path(name)
        target.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        self.add_artefact(target, role or name)
        return target

    def finalise(self, status: str = "ok", note: str | None = None) -> Path:
        self.manifest["status"] = status
        self.manifest["finished_at"] = _utcnow()
        if note:
            self.manifest["note"] = note
        target = self.dir / MANIFEST_NAME
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.manifest, indent=2, default=str), encoding="utf-8")
        _point_latest(self.root / self.kind, self.run_id)
        return target


class Registry:
    """Directory of runs, grouped by kind (`train`, `simulate`, `eval`, ...)."""

    def __init__(self, root: str | os.PathLike[str] = "artefacts") -> None:
        self.root = Path(root)

    def new_run(
        self,
        kind: str,
        config: Any = None,
        run_id: str | None = None,
        tags: Iterable[str] = (),
    ) -> Run:
        rid = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run = Run(run_id=rid, root=self.root, kind=kind)
        run.dir.mkdir(parents=True, exist_ok=True)
        run.manifest = {
            "run_id": rid,
            "kind": kind,
            "started_at": _utcnow(),
            "status": "running",
            "tags": list(tags),
            "environment": {
                "python": sys.version.split()[0],
                "platform": platform.platform(),
                "host": socket.gethostname(),
                "git_rev": _git_rev(),
                "capplan_version": _package_version(),
            },
            "config": config.to_dict() if hasattr(config, "to_dict") else config,
            "artefacts": [],
        }
        return run

    def resolve(self, kind: str, run_id: str | None = None) -> Run:
        """Load an existing run; `None` or 'latest' resolves the latest pointer."""
        base = self.root / kind
        rid = run_id or _LATEST
        if rid == _LATEST:
            rid = read_latest(base)
        run_dir = base / rid
        manifest_path = run_dir / MANIFEST_NAME
        if not manifest_path.exists():
            raise FileNotFoundError(f"no manifest for {kind}/{rid} under {self.root}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        return Run(run_id=rid, root=self.root, kind=kind, manifest=manifest)

    def list_runs(self, kind: str) -> list[str]:
        base = self.root / kind
        if not base.exists():
            return []
        return sorted(p.name for p in base.iterdir() if (p / MANIFEST_NAME).exists())

    def verify(self, kind: str, run_id: str | None = None) -> list[str]:
        """Re-hash artefacts; returns a list of problems (empty means clean)."""
        run = self.resolve(kind, run_id)
        problems: list[str] = []
        for entry in run.manifest.get("artefacts", []):
            path = run.dir / entry["path"]
            if not path.exists():
                problems.append(f"missing: {entry['path']}")
            elif sha256_file(path) != entry["sha256"]:
                problems.append(f"hash mismatch: {entry['path']}")
        return problems


def _point_latest(base: Path, run_id: str) -> None:
    """Record the latest run id.

    A plain text file rather than a symlink: this has to work on the shared
    Windows drive the pack generator writes to.
    """
    base.mkdir(parents=True, exist_ok=True)
    (base / "LATEST").write_text(run_id + "\n", encoding="utf-8")


def read_latest(base: Path) -> str:
    pointer = base / "LATEST"
    if not pointer.exists():
        raise FileNotFoundError(f"no runs recorded under {base}")
    return pointer.read_text(encoding="utf-8").strip()


def _package_version() -> str:
    from capplan import __version__

    return __version__
