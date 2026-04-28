from __future__ import annotations

import json
import os
import platform
import re
import socket
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def capture_run_metadata() -> dict[str, Any]:
    """Capture machine and power context that can materially affect timings."""
    metadata: dict[str, Any] = {
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "cpu_count": os.cpu_count(),
    }
    computer_name = run_text(["scutil", "--get", "ComputerName"])
    if computer_name:
        metadata["computer_name"] = computer_name

    power = capture_power_metadata()
    if power:
        metadata["power"] = power
    return metadata


def capture_power_metadata() -> dict[str, Any]:
    battery_text = run_text(["pmset", "-g", "batt"])
    custom_text = run_text(["pmset", "-g", "custom"])
    power: dict[str, Any] = {}
    if battery_text:
        power["source"] = parse_power_source(battery_text)
        power["battery"] = parse_battery_status(battery_text)
        power["pmset_batt"] = battery_text
    if custom_text:
        power["powermode"] = parse_powermodes(custom_text)
        power["pmset_custom"] = custom_text
    return power


def write_run_metadata(run_dir: Path, metadata: dict[str, Any]) -> Path:
    path = run_dir / "run-metadata.json"
    path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def parse_power_source(text: str) -> str | None:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("Now drawing from"):
            return stripped.removeprefix("Now drawing from").strip().strip("'")
    return None


def parse_battery_status(text: str) -> dict[str, Any]:
    for line in text.splitlines():
        if "-InternalBattery-" not in line:
            continue
        status: dict[str, Any] = {"raw": line.strip()}
        parts = [part.strip() for part in line.split(";")]
        if parts:
            percent = parse_percent(parts[0])
            if percent is not None:
                status["percent"] = percent
        if len(parts) > 1 and parts[1]:
            status["state"] = parts[1]
        if len(parts) > 2 and parts[2]:
            status["time_remaining"] = parts[2]
        return status
    return {}


def parse_percent(text: str) -> int | None:
    match = re.search(r"(\d+)%", text)
    return int(match.group(1)) if match else None


def parse_powermodes(text: str) -> dict[str, int]:
    modes: dict[str, int] = {}
    current_section: str | None = None
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.endswith("Power:"):
            current_section = stripped.removesuffix(":")
            continue
        if current_section and stripped.startswith("powermode"):
            parts = stripped.split()
            if len(parts) >= 2:
                try:
                    modes[current_section] = int(parts[1])
                except ValueError:
                    continue
    return modes


def run_text(command: list[str]) -> str | None:
    try:
        completed = subprocess.run(command, text=True, capture_output=True, check=False, timeout=5)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()
