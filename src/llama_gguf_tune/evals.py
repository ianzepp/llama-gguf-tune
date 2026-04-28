from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class EvalResult:
    run_dir: Path
    model_name: str
    artifact_kind: str
    total_candidates: int
    successful_candidates: int
    failed_candidates: int
    best_generation_tps: float
    best_prompt_tps: float
    best_candidate: dict[str, Any]
    best_runtime_args: list[str]
    run_context: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_dir": str(self.run_dir),
            "model_name": self.model_name,
            "artifact_kind": self.artifact_kind,
            "total_candidates": self.total_candidates,
            "successful_candidates": self.successful_candidates,
            "failed_candidates": self.failed_candidates,
            "best_generation_tps": self.best_generation_tps,
            "best_prompt_tps": self.best_prompt_tps,
            "best_candidate": self.best_candidate,
            "best_runtime_args": self.best_runtime_args,
            "run_context": self.run_context,
        }


def discover_run_dirs(root: Path, artifact_kind: str = "all") -> list[Path]:
    """Return run directories containing benchmark artifacts."""
    patterns = artifact_patterns(artifact_kind)
    if has_run_artifact(root, artifact_kind):
        return [root]
    run_dirs = {
        path.parent
        for pattern in patterns
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


def summarize_runs(run_dirs: list[Path], artifact_kind: str = "auto") -> list[EvalResult]:
    results = [load_eval_result(run_dir, artifact_kind=artifact_kind) for run_dir in run_dirs]
    return sorted(results, key=lambda result: result.best_generation_tps, reverse=True)


def load_eval_result(run_dir: Path, artifact_kind: str = "auto") -> EvalResult:
    path = artifact_path(run_dir, artifact_kind)
    resolved_kind = kind_from_artifact(path)
    records = load_run_records(path)
    if not records:
        raise RuntimeError(f"no benchmark records found in {path}")

    successes = [record for record in records if record_successful(record)]
    if successes:
        best = max(successes, key=record_generation_tps)
        best_generation_tps = record_generation_tps(best)
        best_prompt_tps = record_prompt_tps(best)
        best_candidate = candidate_from_record(best)
        best_runtime_args = runtime_args_from_record(best)
    else:
        best = records[0]
        best_generation_tps = 0.0
        best_prompt_tps = 0.0
        best_candidate = candidate_from_record(best)
        best_runtime_args = runtime_args_from_record(best)

    return EvalResult(
        run_dir=run_dir,
        model_name=model_name_from_record(best, run_dir),
        artifact_kind=resolved_kind,
        total_candidates=len(records),
        successful_candidates=len(successes),
        failed_candidates=len(records) - len(successes),
        best_generation_tps=best_generation_tps,
        best_prompt_tps=best_prompt_tps,
        best_candidate=best_candidate,
        best_runtime_args=best_runtime_args,
        run_context=run_context_from_record(best),
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


def has_run_artifact(run_dir: Path, artifact_kind: str = "all") -> bool:
    return any((run_dir / pattern).is_file() for pattern in artifact_patterns(artifact_kind))


def artifact_path(run_dir: Path, artifact_kind: str = "auto") -> Path:
    if artifact_kind == "server":
        return run_dir / "server.jsonl"
    if artifact_kind == "bench":
        return run_dir / "run.jsonl"
    bench_path = run_dir / "run.jsonl"
    if bench_path.is_file():
        return bench_path
    return run_dir / "server.jsonl"


def artifact_patterns(artifact_kind: str) -> list[str]:
    if artifact_kind in {"all", "auto"}:
        return ["run.jsonl", "server.jsonl"]
    if artifact_kind == "bench":
        return ["run.jsonl"]
    if artifact_kind == "server":
        return ["server.jsonl"]
    raise RuntimeError("--kind must be one of: all, bench, server")


def kind_from_artifact(path: Path) -> str:
    return "server" if path.name == "server.jsonl" else "bench"


def record_successful(record: dict[str, Any]) -> bool:
    if "health_ok" in record or "chat_ok" in record:
        return bool(record.get("health_ok")) and bool(record.get("chat_ok"))
    return record.get("returncode") == 0


def format_eval_table(results: list[EvalResult]) -> str:
    headers = ["kind", "model", "decode tok/s", "prompt tok/s", "success", "failed", "ctx", "best args", "run"]
    rows = [
        [
            result.artifact_kind,
            result.model_name,
            f"{result.best_generation_tps:.3f}",
            f"{result.best_prompt_tps:.3f}",
            f"{result.successful_candidates}/{result.total_candidates}",
            str(result.failed_candidates),
            format_run_context(result.run_context),
            format_best_args(result.best_runtime_args, result.best_candidate),
            str(result.run_dir),
        ]
        for result in results
    ]
    return format_table(headers, rows)


def format_best_args(runtime_args: list[str], candidate: dict[str, Any]) -> str:
    if runtime_args:
        return " ".join(runtime_args)
    return format_candidate(candidate)


def format_candidate(candidate: dict[str, Any]) -> str:
    if not candidate:
        return "-"
    return " ".join(f"{key}={candidate[key]}" for key in sorted(candidate))


def format_run_context(context: dict[str, Any]) -> str:
    power = context.get("power")
    if not isinstance(power, dict):
        return "-"
    source = power.get("source")
    if not isinstance(source, str):
        return "-"
    short_source = "Battery" if source == "Battery Power" else "AC" if source == "AC Power" else source
    powermode = power.get("powermode")
    if isinstance(powermode, dict) and isinstance(powermode.get(source), int):
        return f"{short_source}/{powermode[source]}"
    return short_source


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


def runtime_args_from_record(record: dict[str, Any]) -> list[str]:
    runtime_args = record.get("runtime_args")
    if not isinstance(runtime_args, list):
        return []
    return [str(value) for value in runtime_args]


def run_context_from_record(record: dict[str, Any]) -> dict[str, Any]:
    context = record.get("run")
    return dict(context) if isinstance(context, dict) else {}


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
