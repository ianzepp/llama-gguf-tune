from unittest import TestCase

from llama_gguf_tune.candidates import (
    Candidate,
    build_candidates,
    candidate_from_dict,
    default_candidates,
    drill_candidates,
    select_candidates,
)


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

    def test_candidate_server_args(self) -> None:
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

        self.assertEqual(
            candidate.server_args(),
            [
                "-t",
                "6",
                "-tb",
                "3",
                "-b",
                "1024",
                "-ub",
                "256",
                "-fa",
                "on",
                "-c",
                "4096",
                "-ctk",
                "f16",
                "-ctv",
                "q8_0",
                "--no-mmap",
            ],
        )

    def test_default_candidates_are_nonempty(self) -> None:
        candidates = default_candidates(18)

        self.assertTrue(candidates)
        self.assertTrue(all(candidate.ubatch_size <= candidate.batch_size for candidate in candidates))

    def test_presets_increase_candidate_depth(self) -> None:
        quick = build_candidates(18, preset="quick")
        standard = build_candidates(18, preset="standard")
        thorough = build_candidates(18, preset="thorough")

        self.assertLess(len(quick), len(standard))
        self.assertLess(len(standard), len(thorough))

    def test_select_candidates_deduplicates_by_runtime_kind(self) -> None:
        bench = select_candidates(18, preset="standard", kind="bench", limit=12)
        server = select_candidates(18, preset="standard", kind="server", limit=12)

        self.assertEqual(len({tuple(candidate.bench_args()) for candidate in bench}), len(bench))
        self.assertEqual(len({tuple(candidate.server_args()) for candidate in server}), len(server))

    def test_drill_candidates_start_with_seed_and_mutate_neighbors(self) -> None:
        seed = Candidate(
            threads=9,
            batch_threads=9,
            batch_size=1024,
            ubatch_size=512,
            flash_attn=True,
            mmap=True,
            ctx_size=4096,
            cache_type_k="f16",
            cache_type_v="f16",
        )

        candidates = drill_candidates(seed, 18, limit=10)

        self.assertEqual(candidates[0], seed)
        self.assertEqual(len(candidates), 10)
        self.assertEqual(len({tuple(candidate.server_args()) for candidate in candidates}), len(candidates))
        self.assertTrue(any(candidate.flash_attn is False for candidate in candidates))
        self.assertTrue(all(candidate.ubatch_size <= candidate.batch_size for candidate in candidates))

    def test_candidate_from_dict_rejects_missing_profile_fields(self) -> None:
        with self.assertRaises(RuntimeError):
            candidate_from_dict({"threads": 1})
