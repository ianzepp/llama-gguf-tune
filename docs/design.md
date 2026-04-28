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

## Scoring

The first scoring model prioritizes generation throughput from `llama-bench`.
Future versions should score on:

- generation tokens per second
- prompt tokens per second
- memory pressure
- startup reliability
- server request latency
- quality guardrails

## Safety

The CLI writes artifacts only under the selected runs directory. Promotion to
wrapper scripts or service configs should be a separate explicit command.

