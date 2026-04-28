from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from .bench import create_run_dir, require_llama_bench, run_llama_bench, write_best_profile, write_jsonl
from .candidates import drill_candidates, preset_names, select_candidates
from .drill import latest_server_profile, load_profile_candidate, mark_drill_metadata
from .evals import discover_run_dirs, format_eval_table, select_latest_run_dirs, summarize_runs
from .models import find_gguf_models, human_size
from .run_metadata import capture_run_metadata, write_run_metadata
from .server_eval import aggregate_repetition_results, require_llama_server, run_llama_server_eval


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="llama-gguf-tune",
        description="Autotune llama.cpp runtime flags for local GGUF models.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    scan = subparsers.add_parser("scan", help="find GGUF models under a path")
    scan.add_argument("path", type=Path)
    scan.set_defaults(func=cmd_scan)

    bench = subparsers.add_parser("bench", help="run a llama-bench candidate matrix")
    bench.add_argument("model", type=Path)
    bench.add_argument("--runs-dir", type=Path, default=Path("tuning-runs"))
    bench.add_argument("--llama-bench", dest="llama_bench", default=None)
    bench.add_argument("--limit", type=int, default=12, help="maximum candidates to run")
    bench.add_argument("--preset", choices=preset_names(), default="standard", help="candidate matrix depth")
    bench.add_argument("--repetitions", type=int, default=3)
    bench.add_argument("--prompt-tokens", type=int, default=512)
    bench.add_argument("--gen-tokens", type=int, default=128)
    bench.set_defaults(func=cmd_bench)

    server_eval = subparsers.add_parser("server-eval", help="run temporary llama-server request evals")
    server_eval.add_argument("model", type=Path)
    server_eval.add_argument("--runs-dir", type=Path, default=Path("tuning-runs"))
    server_eval.add_argument("--llama-server", dest="llama_server", default=None)
    server_eval.add_argument("--limit", type=int, default=4, help="maximum candidates to run")
    server_eval.add_argument("--preset", choices=preset_names(), default="standard", help="candidate matrix depth")
    server_eval.add_argument("--repetitions", type=int, default=1, help="requests to run per candidate")
    server_eval.add_argument("--prompt", default="Reply with one concise sentence about local inference tuning.")
    server_eval.add_argument("--max-tokens", type=int, default=64)
    server_eval.add_argument("--startup-timeout", type=float, default=120.0)
    server_eval.add_argument("--request-timeout", type=float, default=60.0)
    server_eval.add_argument("--host", default="127.0.0.1")
    server_eval.add_argument("--port", type=int, default=None)
    server_eval.set_defaults(func=cmd_server_eval)

    drill = subparsers.add_parser("drill", help="refine runtime flags around the latest server winner")
    drill.add_argument("model", type=Path)
    drill.add_argument("--runs-dir", type=Path, default=Path("tuning-runs"))
    drill.add_argument("--llama-server", dest="llama_server", default=None)
    drill.add_argument("--source-profile", type=Path, default=None, help="server-best.json to drill from")
    drill.add_argument("--limit", type=int, default=16, help="maximum neighbor candidates to run")
    drill.add_argument("--repetitions", type=int, default=3, help="requests to run per candidate")
    drill.add_argument("--prompt", default="Reply with one concise sentence about local inference tuning.")
    drill.add_argument("--max-tokens", type=int, default=64)
    drill.add_argument("--startup-timeout", type=float, default=120.0)
    drill.add_argument("--request-timeout", type=float, default=60.0)
    drill.add_argument("--host", default="127.0.0.1")
    drill.add_argument("--port", type=int, default=None)
    drill.set_defaults(func=cmd_drill)

    profile = subparsers.add_parser("profile", help="print the latest saved best profile for a model")
    profile.add_argument("model", type=Path)
    profile.add_argument("--runs-dir", type=Path, default=Path("tuning-runs"))
    profile.set_defaults(func=cmd_profile)

    evals = subparsers.add_parser("eval", help="summarize and rank saved benchmark runs")
    evals.add_argument("runs_dir", type=Path, nargs="?", default=Path("tuning-runs"))
    evals.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    evals.add_argument("--kind", choices=["all", "bench", "server"], default="all")
    evals.add_argument("--latest", action="store_true", help="show only the newest run for each model")
    evals.add_argument("--top", type=int, default=None, help="limit output to the top N runs")
    evals.set_defaults(func=cmd_eval)

    return parser


def cmd_scan(args: argparse.Namespace) -> int:
    models = find_gguf_models(args.path)
    if not models:
        print("No GGUF models found.")
        return 1
    for model in models:
        print(f"{model.path}\t{human_size(model.size_bytes)}")
    return 0


def cmd_bench(args: argparse.Namespace) -> int:
    model = args.model.expanduser().resolve()
    if not model.is_file():
        raise RuntimeError(f"model not found: {model}")
    if model.suffix.lower() != ".gguf":
        raise RuntimeError(f"expected a .gguf model: {model}")

    llama_bench = require_llama_bench(args.llama_bench)
    logical_cpus = os.cpu_count() or 1
    candidates = select_candidates(logical_cpus, preset=args.preset, kind="bench", limit=args.limit)
    if args.repetitions < 1:
        raise RuntimeError("--repetitions must be at least 1")
    run_dir = create_run_dir(args.runs_dir.expanduser().resolve(), model)
    run_metadata = capture_run_metadata()
    write_run_metadata(run_dir, run_metadata)
    print(f"run_dir={run_dir}")
    print(f"preset={args.preset}")
    print(f"candidates={len(candidates)}")
    print(f"repetitions={args.repetitions}")

    results = []
    best = None
    for index, candidate in enumerate(candidates, start=1):
        print(f"[{index}/{len(candidates)}] {candidate.as_dict()}", flush=True)
        result = run_llama_bench(
            model,
            candidate,
            llama_bench=llama_bench,
            repetitions=args.repetitions,
            prompt_tokens=args.prompt_tokens,
            gen_tokens=args.gen_tokens,
            run_metadata=run_metadata,
        )
        results.append(result)
        if result.returncode == 0 and (best is None or result.generation_tps > best.generation_tps):
            best = result
        print(f"  returncode={result.returncode} generation_tps={result.generation_tps:.3f}")

    write_jsonl(run_dir / "run.jsonl", [result.as_dict() for result in results])
    if best is None:
        raise RuntimeError(f"all candidates failed; see {run_dir / 'run.jsonl'}")

    profile_path = write_best_profile(run_dir, model, best, run_metadata)
    print(f"best_generation_tps={best.generation_tps:.3f}")
    print(f"best_profile={profile_path}")
    return 0


def cmd_server_eval(args: argparse.Namespace) -> int:
    model = args.model.expanduser().resolve()
    if not model.is_file():
        raise RuntimeError(f"model not found: {model}")
    if model.suffix.lower() != ".gguf":
        raise RuntimeError(f"expected a .gguf model: {model}")

    llama_server = require_llama_server(args.llama_server)
    logical_cpus = os.cpu_count() or 1
    candidates = select_candidates(logical_cpus, preset=args.preset, kind="server", limit=args.limit)
    if args.repetitions < 1:
        raise RuntimeError("--repetitions must be at least 1")
    run_dir = create_run_dir(args.runs_dir.expanduser().resolve(), model)
    run_metadata = capture_run_metadata()
    write_run_metadata(run_dir, run_metadata)
    print(f"run_dir={run_dir}")
    print(f"preset={args.preset}")
    print(f"candidates={len(candidates)}")
    print(f"repetitions={args.repetitions}")

    results = []
    best = None
    for index, candidate in enumerate(candidates, start=1):
        print(f"[{index}/{len(candidates)}] {candidate.as_dict()}", flush=True)
        repetitions = [
            run_llama_server_eval(
                model,
                candidate,
                llama_server=llama_server,
                prompt=args.prompt,
                max_tokens=args.max_tokens,
                host=args.host,
                port=args.port,
                startup_timeout=args.startup_timeout,
                request_timeout=args.request_timeout,
                run_metadata=run_metadata,
            )
            for _ in range(args.repetitions)
        ]
        result = aggregate_repetition_results(repetitions)
        results.append(result)
        if result.health_ok and result.chat_ok and (best is None or result.generation_tps > best.generation_tps):
            best = result
        print(
            "  "
            f"health_ok={result.health_ok} chat_ok={result.chat_ok} "
            f"generation_tps={result.generation_tps:.3f} latency_seconds={result.latency_seconds:.3f}"
        )

    write_jsonl(run_dir / "server.jsonl", [result.as_dict() for result in results])
    if best is None:
        raise RuntimeError(f"all server candidates failed; see {run_dir / 'server.jsonl'}")

    server_best = run_dir / "server-best.json"
    server_best.write_text(json.dumps(best.as_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"best_generation_tps={best.generation_tps:.3f}")
    print(f"best_profile={server_best}")
    return 0


def cmd_drill(args: argparse.Namespace) -> int:
    model = args.model.expanduser().resolve()
    if not model.is_file():
        raise RuntimeError(f"model not found: {model}")
    if model.suffix.lower() != ".gguf":
        raise RuntimeError(f"expected a .gguf model: {model}")
    if args.repetitions < 1:
        raise RuntimeError("--repetitions must be at least 1")
    if args.limit < 1:
        raise RuntimeError("--limit must be at least 1")

    runs_dir = args.runs_dir.expanduser().resolve()
    source_profile = (
        args.source_profile.expanduser().resolve()
        if args.source_profile is not None
        else latest_server_profile(runs_dir, model)
    )
    seed = load_profile_candidate(source_profile)
    llama_server = require_llama_server(args.llama_server)
    logical_cpus = os.cpu_count() or 1
    candidates = drill_candidates(seed, logical_cpus, limit=args.limit)
    run_dir = create_run_dir(runs_dir, model)
    run_metadata = mark_drill_metadata(
        capture_run_metadata(),
        source_profile=source_profile,
        source_candidate=seed,
    )
    write_run_metadata(run_dir, run_metadata)
    print(f"run_dir={run_dir}")
    print(f"source_profile={source_profile}")
    print(f"seed={seed.as_dict()}")
    print(f"candidates={len(candidates)}")
    print(f"repetitions={args.repetitions}")

    results = []
    best = None
    for index, candidate in enumerate(candidates, start=1):
        print(f"[{index}/{len(candidates)}] {candidate.as_dict()}", flush=True)
        repetitions = [
            run_llama_server_eval(
                model,
                candidate,
                llama_server=llama_server,
                prompt=args.prompt,
                max_tokens=args.max_tokens,
                host=args.host,
                port=args.port,
                startup_timeout=args.startup_timeout,
                request_timeout=args.request_timeout,
                run_metadata=run_metadata,
            )
            for _ in range(args.repetitions)
        ]
        result = aggregate_repetition_results(repetitions)
        results.append(result)
        if result.health_ok and result.chat_ok and (best is None or result.generation_tps > best.generation_tps):
            best = result
        print(
            "  "
            f"health_ok={result.health_ok} chat_ok={result.chat_ok} "
            f"generation_tps={result.generation_tps:.3f} latency_seconds={result.latency_seconds:.3f}"
        )

    write_jsonl(run_dir / "server.jsonl", [result.as_dict() for result in results])
    if best is None:
        raise RuntimeError(f"all drill candidates failed; see {run_dir / 'server.jsonl'}")

    server_best = run_dir / "server-best.json"
    server_best.write_text(json.dumps(best.as_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"best_generation_tps={best.generation_tps:.3f}")
    print(f"best_profile={server_best}")
    return 0


def cmd_profile(args: argparse.Namespace) -> int:
    model = args.model.expanduser().resolve()
    runs_root = args.runs_dir.expanduser().resolve()
    model_dir = runs_root / model.stem
    profiles = sorted(model_dir.glob("*/best.json"))
    if not profiles:
        raise RuntimeError(f"no profiles found under {model_dir}")
    print(profiles[-1].read_text(encoding="utf-8"), end="")
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    runs_dir = args.runs_dir.expanduser().resolve()
    run_dirs = discover_run_dirs(runs_dir, artifact_kind=args.kind)
    if not run_dirs:
        raise RuntimeError(f"no {args.kind} artifacts found under {runs_dir}")
    if args.latest:
        run_dirs = select_latest_run_dirs(run_dirs)

    results = summarize_runs(run_dirs, artifact_kind=args.kind if args.kind != "all" else "auto")
    if args.top is not None:
        if args.top < 1:
            raise RuntimeError("--top must be at least 1")
        results = results[: args.top]
    if args.json:
        print(json.dumps([result.as_dict() for result in results], indent=2, sort_keys=True))
    else:
        print(format_eval_table(results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
