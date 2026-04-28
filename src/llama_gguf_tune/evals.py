from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class EvalResult:
    run_dir: Path
    model_name: str
    total_candidates: int
    successful_candidates: int
    failed_candidates: int
    best_generation_tps: float
    best_prompt_tps: float
    best_candidate: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_dir": str(self.run_dir),
            "model_name": self.model_name,
            "total_candidates": self.total_candidates,
            "successful_candidates": self.successful_candidates,
            "failed_candidates": self.failed_candidates,
            "best_generation_tps": self.best_generation_tps,
            "best_prompt_tps": self.best_prompt_tps,
            "best_candidate": self.best_candidate,
        }


def discover_run_dirs(root: Path) -> list[Path]:
    """Return run directories containing benchmark artifacts."""
    if has_run_artifact(root):
        return [root]
    run_dirs = {
        path.parent
        for pattern in ["run.jsonl", "server.jsonl"]
        for path in root.rglob(pattern)
        if path.is_file()
    }
    return sorted(run_dirs)


def select_latest_run_dirs(run_dirs: list[Path]) -> list[Path]:
    """Keep only the newest run directory for each model directory."""
    latest: dict[Path, Path] = {}
    for run_dir in sorted(run_dirs):
        model_dir = run_dir.parent
        if model_dir not in latest or run_dir.name > latest[model_dir].name:
            latest[model_dir] = run_dir
    return sorted(latest.values(), key=lambda path: path.parent.name)


def summarize_runs(run_dirs: list[Path]) -> list[EvalResult]:
    results = [load_eval_result(run_dir) for run_dir in run_dirs]
    return sorted(results, key=lambda result: result.best_generation_tps, reverse=True)


def load_eval_result(run_dir: Path) -> EvalResult:
    path = artifact_path(run_dir)
    records = load_run_records(path)
    if not records:
        raise RuntimeError(f"no benchmark records found in {path}")

    successes = [record for record in records if record_successful(record)]
    if successes:
        best = max(successes, key=record_generation_tps)
        best_generation_tps = record_generation_tps(best)
        best_prompt_tps = record_prompt_tps(best)
        best_candidate = candidate_from_record(best)
    else:
        best = records[0]
        best_generation_tps = 0.0
        best_prompt_tps = 0.0
        best_candidate = candidate_from_record(best)

    return EvalResult(
        run_dir=run_dir,
        model_name=model_name_from_record(best, run_dir),
        total_candidates=len(records),
        successful_candidates=len(successes),
        failed_candidates=len(records) - len(successes),
        best_generation_tps=best_generation_tps,
        best_prompt_tps=best_prompt_tps,
        best_candidate=best_candidate,
    )


def load_run_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"invalid JSON in {path}:{line_number}: {exc.msg}") from exc
            if not isinstance(payload, dict):
                raise RuntimeError(f"expected object record in {path}:{line_number}")
            records.append(payload)
    return records


def has_run_artifact(run_dir: Path) -> bool:
    return (run_dir / "run.jsonl").is_file() or (run_dir / "server.jsonl").is_file()


def artifact_path(run_dir: Path) -> Path:
    bench_path = run_dir / "run.jsonl"
    if bench_path.is_file():
        return bench_path
    return run_dir / "server.jsonl"


def record_successful(record: dict[str, Any]) -> bool:
    if "health_ok" in record or "chat_ok" in record:
        return bool(record.get("health_ok")) and bool(record.get("chat_ok"))
    return record.get("returncode") == 0


def format_eval_table(results: list[EvalResult]) -> str:
    headers = ["model", "decode tok/s", "prompt tok/s", "success", "failed", "best candidate", "run"]
    rows = [
        [
            result.model_name,
            f"{result.best_generation_tps:.3f}",
            f"{result.best_prompt_tps:.3f}",
            f"{result.successful_candidates}/{result.total_candidates}",
            str(result.failed_candidates),
            format_candidate(result.best_candidate),
            str(result.run_dir),
        ]
        for result in results
    ]
    return format_table(headers, rows)


def format_candidate(candidate: dict[str, Any]) -> str:
    if not candidate:
        return "-"
    return " ".join(f"{key}={candidate[key]}" for key in sorted(candidate))


def format_table(headers: list[str], rows: list[list[str]]) -> str:
    table = [headers, *rows]
    widths = [max(len(row[index]) for row in table) for index in range(len(headers))]
    lines = [format_row(headers, widths), format_row(["-" * width for width in widths], widths)]
    lines.extend(format_row(row, widths) for row in rows)
    return "\n".join(lines)


def format_row(row: list[str], widths: list[int]) -> str:
    return "  ".join(value.ljust(widths[index]) for index, value in enumerate(row))


def record_generation_tps(record: dict[str, Any]) -> float:
    return metric_float(record, "generation_tps")


def record_prompt_tps(record: dict[str, Any]) -> float:
    return metric_float(record, "prompt_tps")


def metric_float(record: dict[str, Any], name: str) -> float:
    metrics = record.get("metrics")
    if not isinstance(metrics, dict):
        return 0.0
    value = metrics.get(name)
    return float(value) if isinstance(value, int | float) else 0.0


def candidate_from_record(record: dict[str, Any]) -> dict[str, Any]:
    candidate = record.get("candidate")
    return dict(candidate) if isinstance(candidate, dict) else {}


def model_name_from_record(record: dict[str, Any], run_dir: Path) -> str:
    metrics = record.get("metrics")
    if isinstance(metrics, dict):
        raw_rows = metrics.get("raw_rows")
        if isinstance(raw_rows, list):
            for row in raw_rows:
                if isinstance(row, dict) and isinstance(row.get("model_filename"), str):
                    return Path(row["model_filename"]).name

    command = record.get("command")
    if isinstance(command, list):
        for index, value in enumerate(command[:-1]):
            if value == "-m" and isinstance(command[index + 1], str):
                return Path(command[index + 1]).name

    return run_dir.parent.name
