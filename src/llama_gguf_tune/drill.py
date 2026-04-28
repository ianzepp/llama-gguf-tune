from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .candidates import Candidate, candidate_from_dict


def latest_server_profile(runs_dir: Path, model_path: Path) -> Path:
    model_dir = runs_dir / model_path.stem
    profiles = sorted(model_dir.glob("*/server-best.json"))
    if not profiles:
        raise RuntimeError(f"no server-best.json profiles found under {model_dir}; run server-eval first")
    return profiles[-1]


def load_profile_candidate(path: Path) -> Candidate:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected object profile in {path}")
    candidate = payload.get("candidate")
    if not isinstance(candidate, dict):
        raise RuntimeError(f"profile has no candidate object: {path}")
    return candidate_from_dict(candidate)


def mark_drill_metadata(
    run_metadata: dict[str, Any],
    *,
    source_profile: Path,
    source_candidate: Candidate,
) -> dict[str, Any]:
    enriched = dict(run_metadata)
    enriched["tuning"] = {
        "mode": "drill",
        "source_profile": str(source_profile),
        "source_candidate": source_candidate.as_dict(),
    }
    return enriched
