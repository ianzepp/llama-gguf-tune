from pathlib import Path
from unittest import TestCase

from llama_gguf_tune.candidates import Candidate
from llama_gguf_tune.server_eval import (
    ServerEvalResult,
    aggregate_repetition_results,
    build_chat_payload,
    build_server_command,
    parse_prometheus_metrics,
)


class ServerEvalTests(TestCase):
    def test_build_server_command_uses_server_flag_spellings(self) -> None:
        candidate = Candidate(
            threads=6,
            batch_threads=3,
            batch_size=1024,
            ubatch_size=256,
            flash_attn=True,
            mmap=False,
            ctx_size=4096,
            cache_type_k="f16",
            cache_type_v="q8_0",
        )

        command = build_server_command(
            llama_server="/usr/local/bin/llama-server",
            model_path=Path("/models/model.gguf"),
            candidate=candidate,
            host="127.0.0.1",
            port=18099,
        )

        self.assertEqual(command[0], "/usr/local/bin/llama-server")
        self.assertIn("--no-mmap", command)
        self.assertNotIn("-mmp", command)
        self.assertIn("-tb", command)
        self.assertIn("3", command)
        self.assertIn("-fa", command)
        self.assertIn("on", command)
        self.assertIn("--parallel", command)
        self.assertIn("--metrics", command)

    def test_parse_prometheus_metrics_extracts_llamacpp_values(self) -> None:
        metrics = parse_prometheus_metrics(
            """
# HELP llamacpp:prompt_tokens_seconds Average prompt throughput in tokens/s.
llamacpp:prompt_tokens_seconds 714.308
llamacpp:predicted_tokens_seconds 23.1343
llamacpp:requests_processing 0
"""
        )

        self.assertEqual(metrics["prompt_tps"], 714.308)
        self.assertEqual(metrics["generation_tps"], 23.1343)
        self.assertEqual(metrics["raw"]["llamacpp:requests_processing"], 0.0)

    def test_build_chat_payload_builds_non_streaming_request(self) -> None:
        payload = build_chat_payload("hello", max_tokens=32)

        self.assertEqual(payload["messages"][0]["role"], "user")
        self.assertEqual(payload["messages"][0]["content"], "hello")
        self.assertEqual(payload["max_tokens"], 32)
        self.assertFalse(payload["stream"])
        self.assertNotIn("stream_options", payload)

    def test_server_eval_result_serializes(self) -> None:
        result = ServerEvalResult(
            candidate={ "threads": 6 },
            command=["llama-server"],
            returncode=0,
            latency_seconds=1.25,
            health_ok=True,
            chat_ok=True,
            metrics={"generation_tps": 42.0},
            response={"id": "chatcmpl"},
            stderr="",
            run_metadata={"power": {"source": "Battery Power"}},
        )

        payload = result.as_dict()

        self.assertEqual(payload["returncode"], 0)
        self.assertEqual(payload["latency_seconds"], 1.25)
        self.assertEqual(payload["metrics"]["generation_tps"], 42.0)
        self.assertEqual(payload["run"]["power"]["source"], "Battery Power")

    def test_aggregate_repetition_results_uses_mean_and_preserves_samples(self) -> None:
        results = [
            ServerEvalResult(
                candidate={"threads": 6},
                command=["llama-server"],
                returncode=0,
                latency_seconds=1.0,
                health_ok=True,
                chat_ok=True,
                metrics={"generation_tps": 10.0, "prompt_tps": 100.0},
                response={"id": "one"},
                stderr="a",
            ),
            ServerEvalResult(
                candidate={"threads": 6},
                command=["llama-server"],
                returncode=0,
                latency_seconds=3.0,
                health_ok=True,
                chat_ok=True,
                metrics={"generation_tps": 20.0, "prompt_tps": 200.0},
                response={"id": "two"},
                stderr="b",
            ),
        ]

        result = aggregate_repetition_results(results)

        self.assertEqual(result.metrics["generation_tps"], 15.0)
        self.assertEqual(result.metrics["generation_tps_min"], 10.0)
        self.assertEqual(result.metrics["generation_tps_max"], 20.0)
        self.assertEqual(result.metrics["prompt_tps"], 150.0)
        self.assertEqual(result.latency_seconds, 2.0)
        self.assertEqual(len(result.metrics["samples"]), 2)
