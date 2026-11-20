"""Build the SHA-256 manifest for the public V4 release surface."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "PUBLIC_MANIFEST_V4.json"
INCLUDE = (
    ROOT / "README.md",
    ROOT / "CHANGELOG_V4.md",
    ROOT / "REPRODUCIBILITY_V4.md",
    ROOT / "THIRD_PARTY_NOTICES.md",
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
    ROOT / "tests" / "v4",
    ROOT / "scripts" / "reproduce_v4.py",
    ROOT / "scripts" / "validate_v4_release.py",
    ROOT / "scripts" / "build_v4_manifest.py",
    ROOT / "scripts" / "export_v4_rule_diagnostics.py",
)


def _iter_files(path: Path):
    if path.is_file():
        yield path
        return
    if path.is_dir():
        for candidate in path.rglob("*"):
            if candidate.is_file() and "__pycache__" not in candidate.parts:
                yield candidate


def main() -> int:
    files = sorted({file for root in INCLUDE for file in _iter_files(root)})
    entries = []
    for path in files:
        data = path.read_bytes()
        entries.append(
            {
                "path": path.relative_to(ROOT).as_posix(),
                "bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        )
    payload = {
        "schema_version": 1,
        "release": "AASM_Rule_V4_Event_to_Stage_Optimization",
        "scope": "Public code and aggregate results; no raw PSG, weights, or record-level traces.",
        "file_count": len(entries),
        "files": entries,
    }
    with OUTPUT.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"output": OUTPUT.name, "file_count": len(entries)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
