# llama-gguf-tune

Inference autotuning for local GGUF models.

`llama-gguf-tune` finds fast, reliable `llama.cpp` runtime flags for GGUF
models on the hardware you actually have. It does not fine-tune model weights.
It tunes inference settings: threads, batch sizes, flash attention, mmap, context
size, KV cache types, and related runtime flags.

## Status

Early scaffold. The first supported workflow is local discovery plus
`llama-bench` matrix runs.

## Why

People download GGUF models and guess the flags:

```sh
llama-server -m model.gguf -ngl 999 -t 8 -b 1024 -ub 256
```

That guess is hardware-specific. The right answer changes across Apple Metal,
Linux CPU, CUDA, ROCm, quantization, context length, batch shape, and model
architecture.

This project makes the guess measurable.

## Install

From a checkout:

```sh
python3 -m pip install -e .
```

For direct local use without installing:

```sh
PYTHONPATH=src python3 -m llama_gguf_tune --help
```

## Usage

Scan a model directory:

```sh
llama-gguf-tune scan /Volumes/ai/models
```

Run a small benchmark matrix:

```sh
llama-gguf-tune bench /Volumes/ai/models/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf
```

Write artifacts to a specific directory:

```sh
llama-gguf-tune bench ./model.gguf --runs-dir ./tuning-runs
```

Print the best saved profile for a model:

```sh
llama-gguf-tune profile ./model.gguf
```

## What Gets Tuned

Initial matrix dimensions:

- `-t`, `--threads`
- `-tb`, batch threads
- `-b`, batch size
- `-ub`, microbatch size
- `-fa`, flash attention
- `-mmp`, mmap
- `-c`, context size
- `-ctk`, KV cache K type
- `-ctv`, KV cache V type

## What Does Not Happen

`llama-gguf-tune` does not train, LoRA-tune, quantize, or modify model weights.
It treats `.gguf` files as immutable inputs and records repeatable runtime
commands plus benchmark results.

## Artifact Layout

Each benchmark run writes JSONL results and a winning profile:

```text
tuning-runs/
  <model-stem>/
    <timestamp>/
      run.jsonl
      best.json
```

The profile contains the model path, runtime flags, benchmark summary, and the
exact command used.

## Roadmap

- `llama-server` request benchmarks against temporary ports.
- Promotion to generated wrapper scripts.
- Multiple hardware presets: Metal, CPU, CUDA, ROCm.
- Draft-model speculative decoding profiles.
- Hugging Face repo and quant discovery.
- Agent loop that proposes bounded candidate matrices.

