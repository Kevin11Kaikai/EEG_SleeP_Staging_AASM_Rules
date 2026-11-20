from __future__ import annotations

import gzip
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from .config import CHAMPION_ROOT, FROZEN_SRC


if str(FROZEN_SRC) not in sys.path:
    sys.path.insert(0, str(FROZEN_SRC))

from aasm_v2.types import EpochEvidence, Evidence  # noqa: E402


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    temporary.replace(path)


def read_json_gz(path: Path) -> Any:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return json.load(stream)


def write_json_gz(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(temporary, "wt", encoding="utf-8", compresslevel=6) as stream:
        json.dump(value, stream, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    temporary.replace(path)


def load_bank(record: str) -> dict:
    return read_json_gz(CHAMPION_ROOT / "event_banks" / f"{record}.json.gz")


def load_sealed_predictions(record: str) -> list[dict]:
    return read_json_gz(CHAMPION_ROOT / "predictions" / f"{record}.json.gz")


def load_rows(record: str) -> tuple[list[EpochEvidence], list[dict]]:
    raw_rows = read_json_gz(CHAMPION_ROOT / "evidence" / f"{record}.json.gz")
    rows = []
    for raw in raw_rows:
        events = {
            key: Evidence(
                seconds=value.get("seconds"),
                event_ids=list(value.get("event_ids", [])),
                detector_status=value.get("detector_status", "AVAILABLE"),
                negative_validated=bool(value.get("negative_validated", False)),
            )
            for key, value in raw["events"].items()
        }
        rows.append(
            EpochEvidence(
                index=int(raw["index"]),
                signals_good=bool(raw["signals_good"]),
                events=events,
                attributes=dict(raw.get("attributes", {})),
            )
        )
    return rows, raw_rows


def artifact_manifest(root: Path) -> list[dict[str, Any]]:
    output = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name == "artifact_manifest.json":
            continue
        output.append(
            {
                "path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return output
