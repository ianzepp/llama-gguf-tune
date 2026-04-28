from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelFile:
    path: Path
    size_bytes: int

    @property
    def stem(self) -> str:
        return self.path.stem


def find_gguf_models(root: Path) -> list[ModelFile]:
    """Return GGUF files under root in stable path order."""
    if root.is_file():
        paths = [root] if root.suffix.lower() == ".gguf" else []
    else:
        paths = sorted(root.rglob("*.gguf"))

    models: list[ModelFile] = []
    for path in paths:
        try:
            stat = path.stat()
        except OSError:
            continue
        if path.is_file():
            models.append(ModelFile(path=path, size_bytes=stat.st_size))
    return models


def human_size(size_bytes: int) -> str:
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    size = float(size_bytes)
    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size_bytes} B"

