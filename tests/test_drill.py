import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from llama_gguf_tune.candidates import Candidate
from llama_gguf_tune.drill import latest_server_profile, load_profile_candidate, mark_drill_metadata


class DrillTests(TestCase):
    def test_latest_server_profile_selects_newest_profile_for_model(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = root / "model.gguf"
            model.write_text("fake", encoding="utf-8")
            old = root / "runs" / "model" / "20260428T000000Z"
            new = root / "runs" / "model" / "20260428T010000Z"
            old.mkdir(parents=True)
            new.mkdir(parents=True)
            (old / "server-best.json").write_text("{}", encoding="utf-8")
            (new / "server-best.json").write_text("{}", encoding="utf-8")

            self.assertEqual(latest_server_profile(root / "runs", model), new / "server-best.json")

    def test_load_profile_candidate_reads_candidate(self) -> None:
        with TemporaryDirectory() as tmp:
            profile = Path(tmp) / "server-best.json"
            profile.write_text(json.dumps({"candidate": candidate_payload()}), encoding="utf-8")

            candidate = load_profile_candidate(profile)

            self.assertEqual(candidate.threads, 9)
            self.assertEqual(candidate.batch_threads, 9)
            self.assertTrue(candidate.flash_attn)

    def test_mark_drill_metadata_preserves_run_metadata_and_source(self) -> None:
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

        metadata = mark_drill_metadata(
            {"power": {"source": "Battery Power"}},
            source_profile=Path("/tmp/server-best.json"),
            source_candidate=seed,
        )

        self.assertEqual(metadata["power"]["source"], "Battery Power")
        self.assertEqual(metadata["tuning"]["mode"], "drill")
        self.assertEqual(metadata["tuning"]["source_candidate"]["threads"], 9)


def candidate_payload() -> dict[str, object]:
    return {
        "threads": 9,
        "batch_threads": 9,
        "batch_size": 1024,
        "ubatch_size": 512,
        "flash_attn": True,
        "mmap": True,
        "ctx_size": 4096,
        "cache_type_k": "f16",
        "cache_type_v": "f16",
    }
