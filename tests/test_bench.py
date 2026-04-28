from unittest import TestCase

from llama_gguf_tune.bench import BenchResult, parse_llama_bench_json
from llama_gguf_tune.candidates import Candidate


class BenchTests(TestCase):
    def test_parse_llama_bench_json_empty_on_invalid_json(self) -> None:
        self.assertEqual(parse_llama_bench_json("not json"), {})


    def test_parse_llama_bench_json_extracts_throughput(self) -> None:
        metrics = parse_llama_bench_json('[{"tg t/s": 12.5, "pp t/s": 99.0}]')

        self.assertEqual(metrics["generation_tps"], 12.5)
        self.assertEqual(metrics["prompt_tps"], 99.0)

    def test_parse_llama_bench_json_extracts_current_avg_ts_shape(self) -> None:
        metrics = parse_llama_bench_json(
            '[{"n_prompt": 64, "n_gen": 0, "avg_ts": 507.6},'
            '{"n_prompt": 0, "n_gen": 16, "avg_ts": 33.3}]'
        )

        self.assertEqual(metrics["prompt_tps"], 507.6)
        self.assertEqual(metrics["generation_tps"], 33.3)

    def test_bench_result_serializes_run_metadata(self) -> None:
        result = BenchResult(
            candidate=Candidate(
                threads=6,
                batch_threads=6,
                batch_size=512,
                ubatch_size=128,
                flash_attn=True,
                mmap=True,
                ctx_size=4096,
                cache_type_k="f16",
                cache_type_v="f16",
            ),
            command=["llama-bench"],
            returncode=0,
            stdout="",
            stderr="",
            metrics={"generation_tps": 12.0},
            run_metadata={"power": {"source": "Battery Power"}},
        )

        self.assertEqual(result.as_dict()["run"]["power"]["source"], "Battery Power")
