from unittest import TestCase

from llama_gguf_tune.bench import parse_llama_bench_json


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
