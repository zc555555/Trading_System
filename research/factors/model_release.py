"""Model release manifest: the seven factor ensembles and factor_weights.json
must come from ONE training run.

train_multi_factor_models.py writes every ensemble to a staging directory,
moves the complete set into place only when all of them exist, then writes
`model_release.json` with the sha256 of every file. Consumers call verify():
a manifest that does not match the files on disk (a crashed retrain left a
mixed set, someone copied one pickle by hand) is refused; a missing manifest
(artifacts from before this check) is reported but tolerated so the nightly
signal run keeps working until the next retrain writes one.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

MANIFEST = "model_release.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ensemble_file(name: str) -> str:
    return f"ensemble_{name}.pkl"


def stage_dir(artifacts_dir: Path) -> Path:
    return Path(artifacts_dir) / ".staging"


def promote(artifacts_dir: Path, names: Iterable[str], extra_files: Iterable[str] = ()) -> None:
    """Move the staged files into place, only when every expected one exists
    in staging (os.replace per file: the window between the first and the
    last move is milliseconds, and the manifest written afterwards lets a
    consumer detect anything that still went wrong)."""
    artifacts_dir, staging = Path(artifacts_dir), stage_dir(artifacts_dir)
    files = [ensemble_file(n) for n in names] + list(extra_files)
    missing = [f for f in files if not (staging / f).exists()]
    if missing:
        raise RuntimeError(f"staging incomplete, nothing promoted: missing {missing}")
    for f in files:
        os.replace(staging / f, artifacts_dir / f)


def write_manifest(artifacts_dir: Path, names: Iterable[str], horizon: Optional[int] = None,
                   data_max_date: Optional[str] = None, extra_files: Iterable[str] = ("factor_weights.json",)) -> dict:
    artifacts_dir = Path(artifacts_dir)
    files = {ensemble_file(n): {"factor": n} for n in names}
    for f in extra_files:
        files[f] = {}
    for f, meta in files.items():
        p = artifacts_dir / f
        if not p.exists():
            raise RuntimeError(f"cannot write manifest: {f} is missing")
        meta["sha256"] = sha256(p)
        meta["bytes"] = p.stat().st_size
    manifest = {"release_id": datetime.now().strftime("%Y%m%dT%H%M%S"),
                "training_date": datetime.now().isoformat(timespec="seconds"),
                "horizon": horizon, "data_max_date": data_max_date, "files": files}
    tmp = artifacts_dir / (MANIFEST + ".tmp")
    tmp.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    os.replace(tmp, artifacts_dir / MANIFEST)
    return manifest


def read_manifest(artifacts_dir: Path) -> Optional[dict]:
    p = Path(artifacts_dir) / MANIFEST
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def verify(artifacts_dir: Path, required: Iterable[str], not_before: Optional[datetime] = None) -> tuple[bool, list[str]]:
    """(ok, messages). ok is False when a manifest exists and any file it
    lists is missing or altered, when a required factor is not in it, or
    when `not_before` is given and the release predates it (a retrain that
    silently left the previous release in place)."""
    artifacts_dir = Path(artifacts_dir)
    required = list(required)
    m = read_manifest(artifacts_dir)
    msgs: list[str] = []
    if m is None:
        msgs.append("no model_release.json yet (pre-manifest artifacts) -- consistency of the model set is UNVERIFIED")
        missing = [n for n in required if not (artifacts_dir / ensemble_file(n)).exists()]
        if missing:
            msgs.append(f"missing ensembles: {missing}")
            return False, msgs
        return True, msgs
    ok = True
    listed = {meta.get("factor") for meta in m["files"].values() if meta.get("factor")}
    for n in required:
        if n not in listed:
            ok = False
            msgs.append(f"factor {n!r} is not part of release {m['release_id']}")
    for f, meta in m["files"].items():
        p = artifacts_dir / f
        if not p.exists():
            ok = False
            msgs.append(f"{f} listed in release {m['release_id']} is missing")
            continue
        if sha256(p) != meta["sha256"]:
            ok = False
            msgs.append(f"{f} differs from release {m['release_id']} (mixed model set)")
    if not_before is not None:
        try:
            trained = datetime.fromisoformat(m["training_date"])
        except Exception:                                # noqa: BLE001
            trained = None
        if trained is None or trained < not_before:
            ok = False
            msgs.append(f"release {m['release_id']} predates this run ({m.get('training_date')} < {not_before.isoformat(timespec='seconds')})")
    if ok:
        msgs.append(f"release {m['release_id']} ({m.get('training_date')}): {len(m['files'])} files verified")
    return ok, msgs
