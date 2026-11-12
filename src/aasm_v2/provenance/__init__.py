"""Stable hashes and atomic local checkpoints."""
import hashlib
import json
from pathlib import Path


def file_hash(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def object_hash(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def write_json(path: str | Path, value: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf8")
    tmp.replace(path)


def source_manifest(root: Path) -> dict:
    return {str(p.relative_to(root)).replace("\\", "/"): file_hash(p)
            for p in sorted((root / "src").rglob("*.py"))}
