from unittest import TestCase

from llama_gguf_tune.candidates import Candidate, default_candidates


class CandidateTests(TestCase):
    def test_candidate_bench_args(self) -> None:
        candidate = Candidate(
            threads=6,
            batch_threads=3,
            batch_size=1024,
            ubatch_size=256,
            flash_attn=True,
            mmap=True,
            ctx_size=4096,
            cache_type_k="f16",
            cache_type_v="f16",
        )

        self.assertEqual(
            candidate.bench_args(),
            [
                "-t",
                "6",
                "-b",
                "1024",
                "-ub",
                "256",
                "-fa",
                "1",
                "-mmp",
                "1",
                "-ctk",
                "f16",
                "-ctv",
                "f16",
            ],
        )

    def test_default_candidates_are_nonempty(self) -> None:
        candidates = default_candidates(18)

        self.assertTrue(candidates)
        self.assertTrue(all(candidate.ubatch_size <= candidate.batch_size for candidate in candidates))
