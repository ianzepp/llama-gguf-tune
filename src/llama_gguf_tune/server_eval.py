from __future__ import annotations

import json
import socket
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from shutil import which
from typing import Any

from .candidates import Candidate


@dataclass(frozen=True)
class ServerEvalResult:
    candidate: dict[str, Any]
    command: list[str]
    returncode: int | None
    latency_seconds: float
    health_ok: bool
    chat_ok: bool
    metrics: dict[str, Any]
    response: dict[str, Any]
    stderr: str

    @property
    def generation_tps(self) -> float:
        return float(self.metrics.get("generation_tps", 0.0) or 0.0)

    def as_dict(self) -> dict[str, Any]:
        return {
            "candidate": self.candidate,
            "command": self.command,
            "returncode": self.returncode,
            "latency_seconds": self.latency_seconds,
            "health_ok": self.health_ok,
            "chat_ok": self.chat_ok,
            "metrics": self.metrics,
            "response": self.response,
            "stderr": self.stderr,
        }


def require_llama_server(binary: str | None = None) -> str:
    resolved = binary or which("llama-server")
    if not resolved:
        raise RuntimeError("llama-server not found on PATH; pass --llama-server /path/to/llama-server")
    return resolved


def find_free_port(host: str = "127.0.0.1") -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])


def build_server_command(
    *,
    llama_server: str,
    model_path: Path,
    candidate: Candidate,
    host: str,
    port: int,
) -> list[str]:
    command = [
        llama_server,
        "-m",
        str(model_path),
        "-t",
        str(candidate.threads),
        "-tb",
        str(candidate.batch_threads),
        "-b",
        str(candidate.batch_size),
        "-ub",
        str(candidate.ubatch_size),
        "-fa",
        "on" if candidate.flash_attn else "off",
        "-c",
        str(candidate.ctx_size),
        "-ctk",
        candidate.cache_type_k,
        "-ctv",
        candidate.cache_type_v,
        "--host",
        host,
        "--port",
        str(port),
        "--parallel",
        "1",
        "--metrics",
    ]
    command.append("--mmap" if candidate.mmap else "--no-mmap")
    return command


def build_chat_payload(prompt: str, *, max_tokens: int) -> dict[str, Any]:
    return {
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": False,
    }


def run_llama_server_eval(
    model_path: Path,
    candidate: Candidate,
    *,
    llama_server: str,
    prompt: str,
    max_tokens: int,
    host: str = "127.0.0.1",
    port: int | None = None,
    startup_timeout: float = 120.0,
    request_timeout: float = 60.0,
) -> ServerEvalResult:
    selected_port = port if port is not None else find_free_port(host)
    command = build_server_command(
        llama_server=llama_server,
        model_path=model_path,
        candidate=candidate,
        host=host,
        port=selected_port,
    )
    started = time.monotonic()
    stderr_file = tempfile.TemporaryFile(mode="w+", encoding="utf-8")
    process = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=stderr_file, text=True)
    stderr = ""
    health_ok = False
    chat_ok = False
    response: dict[str, Any] = {}
    metrics: dict[str, Any] = {}
    try:
        health_ok = wait_for_health(host, selected_port, startup_timeout)
        if health_ok:
            response = post_chat_completion(
                host,
                selected_port,
                build_chat_payload(prompt, max_tokens=max_tokens),
                timeout=request_timeout,
            )
            chat_ok = True
            metrics = parse_prometheus_metrics(fetch_text(host, selected_port, "/metrics", timeout=request_timeout))
        returncode = process.poll()
    except Exception as exc:
        response = {"error": str(exc)}
        returncode = process.poll()
    finally:
        stderr = terminate_process(process, stderr_file)
        returncode = process.returncode
        stderr_file.close()

    return ServerEvalResult(
        candidate=candidate.as_dict(),
        command=command,
        returncode=returncode,
        latency_seconds=time.monotonic() - started,
        health_ok=health_ok,
        chat_ok=chat_ok,
        metrics=metrics,
        response=response,
        stderr=stderr,
    )


def wait_for_health(host: str, port: int, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            payload = fetch_json(host, port, "/health", timeout=2.0)
        except (OSError, urllib.error.URLError, json.JSONDecodeError):
            time.sleep(0.25)
            continue
        if payload.get("status") == "ok":
            return True
        time.sleep(0.25)
    return False


def post_chat_completion(host: str, port: int, payload: dict[str, Any], *, timeout: float) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        f"http://{host}:{port}/v1/chat/completions",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def fetch_json(host: str, port: int, path: str, *, timeout: float) -> dict[str, Any]:
    return json.loads(fetch_text(host, port, path, timeout=timeout))


def fetch_text(host: str, port: int, path: str, *, timeout: float) -> str:
    with urllib.request.urlopen(f"http://{host}:{port}{path}", timeout=timeout) as response:
        return response.read().decode("utf-8")


def parse_prometheus_metrics(text: str) -> dict[str, Any]:
    raw: dict[str, float] = {}
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split()
        if len(parts) < 2:
            continue
        try:
            raw[parts[0]] = float(parts[1])
        except ValueError:
            continue

    metrics: dict[str, Any] = {"raw": raw}
    if "llamacpp:prompt_tokens_seconds" in raw:
        metrics["prompt_tps"] = raw["llamacpp:prompt_tokens_seconds"]
    if "llamacpp:predicted_tokens_seconds" in raw:
        metrics["generation_tps"] = raw["llamacpp:predicted_tokens_seconds"]
    return metrics


def terminate_process(process: subprocess.Popen[str], stderr_file: Any) -> str:
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)
    stderr_file.seek(0)
    return stderr_file.read() or ""
