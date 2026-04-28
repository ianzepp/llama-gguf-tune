import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from llama_gguf_tune.evals import (
    EvalResult,
    discover_run_dirs,
    format_eval_table,
    load_eval_result,
    select_latest_run_dirs,
    summarize_runs,
)


class EvalTests(TestCase):
    def test_load_eval_result_selects_best_successful_generation_score(self) -> None:
        with TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            write_jsonl(
                run_dir / "run.jsonl",
                [
                    record("slow.gguf", 0, 10.0, 50.0, {"threads": 4}),
                    record("slow.gguf", 1, 999.0, 999.0, {"threads": 99}),
                    record("slow.gguf", 0, 12.5, 45.0, {"threads": 6}),
                ],
            )

            result = load_eval_result(run_dir)

            self.assertEqual(result.run_dir, run_dir)
            self.assertEqual(result.model_name, "slow.gguf")
            self.assertEqual(result.total_candidates, 3)
            self.assertEqual(result.successful_candidates, 2)
            self.assertEqual(result.failed_candidates, 1)
            self.assertEqual(result.best_generation_tps, 12.5)
            self.assertEqual(result.best_prompt_tps, 45.0)
            self.assertEqual(result.best_candidate["threads"], 6)

    def test_discover_run_dirs_returns_only_dirs_with_run_jsonl(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            good = root / "model-a" / "20260428T000000Z"
            good.mkdir(parents=True)
            (good / "run.jsonl").write_text("{}", encoding="utf-8")
            server = root / "model-c" / "20260428T010000Z"
            server.mkdir(parents=True)
            (server / "server.jsonl").write_text("{}", encoding="utf-8")
            ignored = root / "model-b"
            ignored.mkdir()

            self.assertEqual(discover_run_dirs(root), [good, server])

    def test_load_eval_result_reads_server_jsonl(self) -> None:
        with TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            write_jsonl(
                run_dir / "server.jsonl",
                [
                    server_record("server.gguf", True, True, 20.0, 100.0, {"threads": 4}),
                    server_record("server.gguf", True, False, 999.0, 999.0, {"threads": 99}),
                    server_record("server.gguf", True, True, 25.0, 90.0, {"threads": 6}),
                ],
            )

            result = load_eval_result(run_dir)

            self.assertEqual(result.model_name, "server.gguf")
            self.assertEqual(result.total_candidates, 3)
            self.assertEqual(result.successful_candidates, 2)
            self.assertEqual(result.failed_candidates, 1)
            self.assertEqual(result.best_generation_tps, 25.0)
            self.assertEqual(result.best_candidate["threads"], 6)

    def test_select_latest_run_dirs_keeps_newest_run_for_each_model(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            old = root / "model-a" / "20260428T000000Z"
            new = root / "model-a" / "20260428T010000Z"
            other = root / "model-b" / "20260428T003000Z"
            for run_dir in [old, new, other]:
                run_dir.mkdir(parents=True)
                (run_dir / "run.jsonl").write_text("{}", encoding="utf-8")

            self.assertEqual(select_latest_run_dirs([old, other, new]), [new, other])

    def test_summarize_runs_sorts_by_generation_score_descending(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            fast = make_run(root, "fast", 40.0)
            slow = make_run(root, "slow", 10.0)

            results = summarize_runs([slow, fast])

            self.assertEqual([result.model_name for result in results], ["fast.gguf", "slow.gguf"])

    def test_format_eval_table_includes_failures_and_best_candidate(self) -> None:
        result = EvalResult(
            run_dir=Path("/tmp/run"),
            model_name="model.gguf",
            total_candidates=3,
            successful_candidates=2,
            failed_candidates=1,
            best_generation_tps=12.3456,
            best_prompt_tps=99.9,
            best_candidate={"threads": 6, "flash_attn": True},
        )

        table = format_eval_table([result])

        self.assertIn("model.gguf", table)
        self.assertIn("12.346", table)
        self.assertIn("2/3", table)
        self.assertIn("1", table)
        self.assertIn("threads=6", table)
        self.assertIn("flash_attn=True", table)


def make_run(root: Path, model_stem: str, generation_tps: float) -> Path:
    run_dir = root / model_stem / "20260428T000000Z"
    run_dir.mkdir(parents=True)
    write_jsonl(run_dir / "run.jsonl", [record(f"{model_stem}.gguf", 0, generation_tps, 100.0, {})])
    return run_dir


def write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for item in records:
            handle.write(json.dumps(item) + "\n")


def record(
    model: str,
    returncode: int,
    generation_tps: float,
    prompt_tps: float,
    candidate: dict[str, object],
) -> dict[str, object]:
    return {
        "command": ["llama-bench", "-m", model],
        "returncode": returncode,
        "candidate": candidate,
        "metrics": {
            "generation_tps": generation_tps,
            "prompt_tps": prompt_tps,
            "raw_rows": [{"model_filename": model}],
        },
    }


def server_record(
    model: str,
    health_ok: bool,
    chat_ok: bool,
    generation_tps: float,
    prompt_tps: float,
    candidate: dict[str, object],
) -> dict[str, object]:
    return {
        "command": ["llama-server", "-m", model],
        "returncode": -15,
        "health_ok": health_ok,
        "chat_ok": chat_ok,
        "candidate": candidate,
        "metrics": {
            "generation_tps": generation_tps,
            "prompt_tps": prompt_tps,
        },
    }
