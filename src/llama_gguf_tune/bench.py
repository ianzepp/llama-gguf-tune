from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from shutil import which
from typing import Any

from .candidates import Candidate


@dataclass(frozen=True)
class BenchResult:
    candidate: Candidate
    command: list[str]
    returncode: int
    stdout: str
    stderr: str
    metrics: dict[str, Any]

    @property
    def generation_tps(self) -> float:
        return float(self.metrics.get("generation_tps", 0.0) or 0.0)

    def as_dict(self) -> dict[str, Any]:
        return {
            "candidate": self.candidate.as_dict(),
            "command": self.command,
            "returncode": self.returncode,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "metrics": self.metrics,
        }


def require_llama_bench(binary: str | None = None) -> str:
    resolved = binary or which("llama-bench")
    if not resolved:
        raise RuntimeError("llama-bench not found on PATH; pass --llama-bench /path/to/llama-bench")
    return resolved


def run_llama_bench(
    model_path: Path,
    candidate: Candidate,
    *,
    llama_bench: str,
    repetitions: int,
    prompt_tokens: int,
    gen_tokens: int,
) -> BenchResult:
    command = [
        llama_bench,
        "-m",
        str(model_path),
        "-r",
        str(repetitions),
        "-p",
        str(prompt_tokens),
        "-n",
        str(gen_tokens),
        "-o",
        "json",
        *candidate.bench_args(),
    ]
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    metrics = parse_llama_bench_json(completed.stdout)
    return BenchResult(
        candidate=candidate,
        command=command,
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
        metrics=metrics,
    )


def parse_llama_bench_json(stdout: str) -> dict[str, Any]:
    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError:
        return {}

    rows = payload if isinstance(payload, list) else [payload]
    generation_values: list[float] = []
    prompt_values: list[float] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        for key, value in row.items():
            lower = key.lower()
            if isinstance(value, int | float) and ("tg" in lower or "gen" in lower) and "t/s" in lower:
                generation_values.append(float(value))
            if isinstance(value, int | float) and ("pp" in lower or "prompt" in lower) and "t/s" in lower:
                prompt_values.append(float(value))
        if isinstance(row.get("avg_ts"), int | float):
            if row.get("n_gen", 0):
                generation_values.append(float(row["avg_ts"]))
            if row.get("n_prompt", 0):
                prompt_values.append(float(row["avg_ts"]))

    metrics: dict[str, Any] = {"raw_rows": rows}
    if generation_values:
        metrics["generation_tps"] = max(generation_values)
    if prompt_values:
        metrics["prompt_tps"] = max(prompt_values)
    return metrics


def create_run_dir(runs_dir: Path, model_path: Path) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe_stem = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in model_path.stem)
    run_dir = runs_dir / safe_stem / timestamp
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def write_best_profile(run_dir: Path, model_path: Path, best: BenchResult) -> Path:
    profile_path = run_dir / "best.json"
    profile = {
        "model": str(model_path),
        "candidate": best.candidate.as_dict(),
        "metrics": best.metrics,
        "command": best.command,
        "environment": {
            "platform": os.uname().sysname,
            "machine": os.uname().machine,
            "release": os.uname().release,
        },
    }
    profile_path.write_text(json.dumps(profile, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return profile_path
