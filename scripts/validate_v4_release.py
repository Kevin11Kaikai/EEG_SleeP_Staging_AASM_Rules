"""Validate that the V4 public-release surface is portable and aggregate-only."""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aasm_v4.contracts import validate_handler_registry  # noqa: E402


TEXT_SUFFIXES = {".csv", ".json", ".md", ".py", ".toml", ".yaml", ".yml", ".txt"}
FORBIDDEN_SUFFIXES = {
    ".bdf",
    ".ckpt",
    ".edf",
    ".npz",
    ".onnx",
    ".pt",
    ".pth",
    ".safetensors",
    ".sqlite",
}
PUBLIC_SCOPES = (
    ROOT / "configs" / "v4",
    ROOT / "src" / "aasm_v4",
    ROOT / "experiments" / "v4_handler_validation",
    ROOT / "experiments" / "v4_36arm_ablation",
    ROOT / "experiments" / "v4_subset_search",
    ROOT / "experiments" / "v4_exhaustive_search",
    ROOT / "experiments" / "v4_mechanistic_attribution",
    ROOT / "experiments" / "v4_rule_discovery",
    ROOT / "results" / "release" / "v4",
    ROOT / "reports" / "v4",
    ROOT / "educational_images" / "v4",
    ROOT / "model_cards",
    ROOT / "provenance" / "v4",
)
RECORD_LEVEL_SCOPES = (
    ROOT / "configs" / "v4",
    ROOT / "results" / "release" / "v4",
    ROOT / "reports" / "v4",
    ROOT / "educational_images" / "v4",
    ROOT / "model_cards",
    ROOT / "provenance" / "v4",
)


def _files(scopes: Iterable[Path]) -> Iterable[Path]:
    seen: set[Path] = set()
    for scope in scopes:
        if not scope.exists():
            continue
        for path in scope.rglob("*"):
            if path.is_file() and path not in seen:
                seen.add(path)
                yield path


def _relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def validate() -> dict[str, object]:
    errors: list[str] = []
    files = list(_files(PUBLIC_SCOPES))
    for path in files:
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            errors.append(f"forbidden binary artifact: {_relative(path)}")
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        text = _read_text(path)
        if re.search(r"(?i)(?:[A-Z]:\\(?:Users|EEG[_-]?Conformer)\\|/home/|/Users/)", text):
            errors.append(f"machine-specific absolute path: {_relative(path)}")
        if re.search(r"(?i)(api[_-]?key|password|bearer\s+[A-Za-z0-9._-]+|private key)", text):
            errors.append(f"possible credential material: {_relative(path)}")

    for path in _files(RECORD_LEVEL_SCOPES):
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        text = _read_text(path)
        if re.search(r"\bSN(?:00[1-9]|010)\b", text):
            errors.append(f"record-level identifier in public aggregate scope: {_relative(path)}")
        if re.search(r"(?i)\b(per[_-]?night|first_divergence_trace|changed_epoch_trace)\b", path.name):
            errors.append(f"record-level artifact name: {_relative(path)}")

    handler_path = ROOT / "configs" / "v4" / "handlers.json"
    with handler_path.open("r", encoding="utf-8") as stream:
        handler_payload = json.load(stream)
    validate_handler_registry(handler_payload["handlers"])

    manifest_path = ROOT / "PUBLIC_MANIFEST_V4.json"
    manifest_checked = False
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for item in manifest.get("files", []):
            path = ROOT / item["path"]
            if not path.is_file():
                errors.append(f"manifest entry missing: {item['path']}")
                continue
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest != item["sha256"]:
                errors.append(f"manifest hash mismatch: {item['path']}")
        manifest_checked = True

    result: dict[str, object] = {
        "status": "PASS" if not errors else "FAIL",
        "scanned_file_count": len(files),
        "manifest_checked": manifest_checked,
        "errors": errors,
    }
    return result


def main() -> int:
    result = validate()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
