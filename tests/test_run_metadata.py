from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from llama_gguf_tune.run_metadata import (
    parse_battery_status,
    parse_power_source,
    parse_powermodes,
    write_run_metadata,
)


class RunMetadataTests(TestCase):
    def test_parse_power_source(self) -> None:
        source = parse_power_source("Now drawing from 'Battery Power'\n")

        self.assertEqual(source, "Battery Power")

    def test_parse_battery_status(self) -> None:
        status = parse_battery_status(
            " -InternalBattery-0 (id=5439587)\t79%; discharging; 8:22 remaining present: true"
        )

        self.assertEqual(status["percent"], 79)
        self.assertEqual(status["state"], "discharging")
        self.assertEqual(status["time_remaining"], "8:22 remaining present: true")

    def test_parse_powermodes(self) -> None:
        modes = parse_powermodes(
            """
Battery Power:
 powermode            1
AC Power:
 powermode            0
"""
        )

        self.assertEqual(modes, {"Battery Power": 1, "AC Power": 0})

    def test_write_run_metadata(self) -> None:
        with TemporaryDirectory() as tmp:
            path = write_run_metadata(Path(tmp), {"power": {"source": "Battery Power"}})

            self.assertTrue(path.is_file())
            self.assertIn("Battery Power", path.read_text(encoding="utf-8"))

