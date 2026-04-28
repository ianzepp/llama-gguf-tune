import contextlib
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

from llama_gguf_tune.cli import main
from llama_gguf_tune.server_eval import ServerEvalResult


class CliTests(TestCase):
    def test_eval_command_prints_ranked_table(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_run(root, "small", 123.4)

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                exit_code = main(["eval", str(root)])

            self.assertEqual(exit_code, 0)
            self.assertIn("small.gguf", stdout.getvalue())
            self.assertIn("123.400", stdout.getvalue())

    def test_eval_command_prints_json(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_run(root, "small", 123.4)

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                exit_code = main(["eval", str(root), "--json"])

            self.assertEqual(exit_code, 0)
            payload = json.loads(stdout.getvalue())
            self.assertEqual(payload[0]["model_name"], "small.gguf")
            self.assertEqual(payload[0]["best_generation_tps"], 123.4)

    def test_eval_command_latest_and_top_filters_results(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_run(root, "same", 10.0, timestamp="20260428T000000Z")
            make_run(root, "same", 20.0, timestamp="20260428T010000Z")
            make_run(root, "other", 30.0, timestamp="20260428T000000Z")

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                exit_code = main(["eval", str(root), "--latest", "--top", "1", "--json"])

            self.assertEqual(exit_code, 0)
            payload = json.loads(stdout.getvalue())
            self.assertEqual(len(payload), 1)
            self.assertEqual(payload[0]["model_name"], "other.gguf")
            self.assertEqual(payload[0]["best_generation_tps"], 30.0)

    def test_server_eval_command_writes_server_artifacts(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = root / "model.gguf"
            model.write_text("fake", encoding="utf-8")
            result = ServerEvalResult(
                candidate={"threads": 6},
                command=["llama-server", "-m", str(model)],
                returncode=-15,
                latency_seconds=1.0,
                health_ok=True,
                chat_ok=True,
                metrics={"generation_tps": 42.0, "prompt_tps": 100.0},
                response={"id": "chatcmpl"},
                stderr="",
            )

            stdout = io.StringIO()
            with (
                patch("llama_gguf_tune.cli.require_llama_server", return_value="/bin/llama-server"),
                patch("llama_gguf_tune.cli.run_llama_server_eval", return_value=result),
                contextlib.redirect_stdout(stdout),
            ):
                exit_code = main(["server-eval", str(model), "--runs-dir", str(root / "runs"), "--limit", "1"])

            self.assertEqual(exit_code, 0)
            self.assertIn("best_generation_tps=42.000", stdout.getvalue())
            self.assertEqual(len(list((root / "runs").rglob("server.jsonl"))), 1)
            self.assertEqual(len(list((root / "runs").rglob("server-best.json"))), 1)


def make_run(root: Path, model_stem: str, generation_tps: float, timestamp: str = "20260428T000000Z") -> None:
    run_dir = root / model_stem / timestamp
    run_dir.mkdir(parents=True)
    record = {
        "command": ["llama-bench", "-m", f"{model_stem}.gguf"],
        "returncode": 0,
        "candidate": {"threads": 6},
        "metrics": {"generation_tps": generation_tps, "prompt_tps": 200.0},
    }
    (run_dir / "run.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")
