"""Build the public V3 artifact manifest without including the manifest itself."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "PUBLIC_MANIFEST.json"
BASE_COMMIT = "155baa7895c2a4a20f8a788901f8412c3f8780b3"


def git_lines(*args: str) -> list[str]:
    result = subprocess.run(
        ["git", *args], cwd=ROOT, text=True, capture_output=True, check=True
    )
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    tracked = set(git_lines("ls-files"))
    untracked = set(git_lines("ls-files", "--others", "--exclude-standard"))
    names = sorted((tracked | untracked) - {"PUBLIC_MANIFEST.json"})
    files = {
        name: {"sha256": sha256(ROOT / name), "bytes": (ROOT / name).stat().st_size}
        for name in names
        if (ROOT / name).is_file()
    }
    evidence = json.loads(
        (ROOT / "reports" / "v3_evidence_summary.json").read_text(encoding="utf-8")
    )
    branch = git_lines("branch", "--show-current")[0]
    core_changed = (
        subprocess.run(
            ["git", "diff", "--quiet", BASE_COMMIT, "--", "src", "configs"],
            cwd=ROOT,
            check=False,
        ).returncode
        != 0
    )
    payload = {
        "schema_version": 3,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "branch": branch,
        "base_branch": "AASM_Rules_V2",
        "base_commit": BASE_COMMIT,
        "status": "V3_RESEARCH_RELEASE_CANDIDATE",
        "raw_or_epoch_clinical_content_included": False,
        "frozen_core_or_configs_modified": core_changed,
        "file_count_excluding_manifest": len(files),
        "bytes_excluding_manifest": sum(item["bytes"] for item in files.values()),
        "release_scope": {
            "included": evidence["public_release_boundary"]["included"],
            "excluded": evidence["public_release_boundary"]["excluded"],
        },
        "sealed_source_artifacts": {
            key: {"sha256": value["sha256"], "bytes": value["bytes"]}
            for key, value in evidence["source_artifacts"].items()
        },
        "files": files,
    }
    with OUTPUT.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(
        f"wrote {OUTPUT.name}: {len(files)} files, {payload['bytes_excluding_manifest']} bytes"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
