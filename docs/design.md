# Design

`llama-gguf-tune` tunes immutable GGUF model files by searching runtime flags.

## Non-Goals

- No model weight training.
- No quantization pipeline in v0.
- No system service mutation during benchmark runs.
- No unbounded agentic shell access.

## Core Concepts

### Model

A local `.gguf` file. The model path is treated as the stable input.

### Candidate

A set of `llama.cpp` flags to test.

### Run

One benchmark execution for one model and one candidate.

### Profile

The selected best candidate plus benchmark evidence.

### Eval Summary

A comparison view over saved run artifacts. Eval summaries are intentionally
derived from `run.jsonl` rather than live benchmark state, so users can compare
old runs, publish evidence, and audit the selected profile later.

## Scoring

The first scoring model prioritizes generation throughput from `llama-bench`.
Future versions should score on:

- generation tokens per second
- prompt tokens per second
- memory pressure
- startup reliability
- server request latency
- quality guardrails

The current `eval` command ranks saved runs by best successful generation
throughput and reports the winning candidate, prompt throughput, success count,
and failure count.

## Safety

The CLI writes artifacts only under the selected runs directory. Promotion to
wrapper scripts or service configs should be a separate explicit command.
