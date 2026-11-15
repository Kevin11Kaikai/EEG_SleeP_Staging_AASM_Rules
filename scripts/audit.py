"""Verify frozen scientific core and optional public-file manifest."""

from _common import ROOT, check_core, read_json
from aasm_v2.provenance import file_hash


if __name__ == "__main__":
    print("Core verified", check_core()["source_config_test_hash"])
    manifest = ROOT / "PUBLIC_MANIFEST.json"
    if manifest.exists():
        data = read_json(manifest)
        for name, entry in data["files"].items():
            digest = entry["sha256"] if isinstance(entry, dict) else entry
            assert file_hash(ROOT / name) == digest, name
        print("Public manifest verified", len(data["files"]), "files")
