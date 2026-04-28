from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from .bench import create_run_dir, require_llama_bench, run_llama_bench, write_best_profile, write_jsonl
from .candidates import default_candidates
from .models import find_gguf_models, human_size


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
    bench.add_argument("--repetitions", type=int, default=3)
    bench.add_argument("--prompt-tokens", type=int, default=512)
    bench.add_argument("--gen-tokens", type=int, default=128)
    bench.set_defaults(func=cmd_bench)

    profile = subparsers.add_parser("profile", help="print the latest saved best profile for a model")
    profile.add_argument("model", type=Path)
    profile.add_argument("--runs-dir", type=Path, default=Path("tuning-runs"))
    profile.set_defaults(func=cmd_profile)

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
    candidates = default_candidates(logical_cpus)[: args.limit]
    run_dir = create_run_dir(args.runs_dir.expanduser().resolve(), model)
    print(f"run_dir={run_dir}")
    print(f"candidates={len(candidates)}")

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
        )
        results.append(result)
        if result.returncode == 0 and (best is None or result.generation_tps > best.generation_tps):
            best = result
        print(f"  returncode={result.returncode} generation_tps={result.generation_tps:.3f}")

    write_jsonl(run_dir / "run.jsonl", [result.as_dict() for result in results])
    if best is None:
        raise RuntimeError(f"all candidates failed; see {run_dir / 'run.jsonl'}")

    profile_path = write_best_profile(run_dir, model, best)
    print(f"best_generation_tps={best.generation_tps:.3f}")
    print(f"best_profile={profile_path}")
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


if __name__ == "__main__":
    raise SystemExit(main())

