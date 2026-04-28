from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from llama_gguf_tune.models import find_gguf_models, human_size


class ModelTests(TestCase):
    def test_find_gguf_models_sorts_paths(self) -> None:
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            nested = tmp_path / "nested"
            nested.mkdir()
            later = nested / "z.gguf"
            earlier = tmp_path / "a.gguf"
            ignored = tmp_path / "notes.txt"
            later.write_text("z", encoding="utf-8")
            earlier.write_text("a", encoding="utf-8")
            ignored.write_text("nope", encoding="utf-8")

            models = find_gguf_models(tmp_path)

            self.assertEqual([model.path for model in models], [earlier, later])

    def test_human_size(self) -> None:
        self.assertEqual(human_size(1), "1.0 B")
        self.assertEqual(human_size(1024), "1.0 KiB")
