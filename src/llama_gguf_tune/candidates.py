from __future__ import annotations

from dataclasses import dataclass
from itertools import product


@dataclass(frozen=True)
class Candidate:
    threads: int
    batch_threads: int
    batch_size: int
    ubatch_size: int
    flash_attn: bool
    mmap: bool
    ctx_size: int
    cache_type_k: str
    cache_type_v: str

    def bench_args(self) -> list[str]:
        return [
            "-t",
            str(self.threads),
            "-b",
            str(self.batch_size),
            "-ub",
            str(self.ubatch_size),
            "-fa",
            "1" if self.flash_attn else "0",
            "-mmp",
            "1" if self.mmap else "0",
            "-ctk",
            self.cache_type_k,
            "-ctv",
            self.cache_type_v,
        ]

    def as_dict(self) -> dict[str, object]:
        return {
            "threads": self.threads,
            "batch_threads": self.batch_threads,
            "batch_size": self.batch_size,
            "ubatch_size": self.ubatch_size,
            "flash_attn": self.flash_attn,
            "mmap": self.mmap,
            "ctx_size": self.ctx_size,
            "cache_type_k": self.cache_type_k,
            "cache_type_v": self.cache_type_v,
        }


def default_candidates(logical_cpus: int) -> list[Candidate]:
    """Build a conservative first-pass candidate matrix."""
    thread_options = sorted({max(1, logical_cpus // 3), max(1, logical_cpus // 2), logical_cpus})
    batch_thread_options = sorted({max(1, logical_cpus // 3), max(1, logical_cpus // 2)})
    candidates: list[Candidate] = []

    for threads, batch_threads, batch_size, ubatch_size, flash_attn, mmap in product(
        thread_options,
        batch_thread_options,
        [512, 1024, 2048],
        [128, 256, 512],
        [False, True],
        [True],
    ):
        if ubatch_size > batch_size:
            continue
        candidates.append(
            Candidate(
                threads=threads,
                batch_threads=batch_threads,
                batch_size=batch_size,
                ubatch_size=ubatch_size,
                flash_attn=flash_attn,
                mmap=mmap,
                ctx_size=4096,
                cache_type_k="f16",
                cache_type_v="f16",
            )
        )

    return candidates
