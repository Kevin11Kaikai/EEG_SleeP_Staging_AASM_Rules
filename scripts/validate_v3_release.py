"""Validate scientific, visual, link, privacy, and frozen-heritage release gates."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import struct
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BASE_COMMIT = "155baa7895c2a4a20f8a788901f8412c3f8780b3"
TARGET_BRANCH = "AASM_Rule_V3_Event_Detectors_Improvement"
PNG_SIZE = (1800, 2400)
SVG_VIEWBOX = "0 0 864 1152"
EVIDENCE_SHA256 = "64d6bcd3de710bd309d92f3835d973900b3ee538a738a5280385eb9fec92f98f"
STEMS = [
    "v3_01_events_to_stages_pass1_pass2",
    "v3_02_spindle_yasa_wonambi",
    "v3_03_kcomplex_mtkcd",
    "v3_04_arousal_unet",
    "v3_05_rem_antiphase",
    "v3_06_event_to_stage_ablation",
    "v3_07_remaining_bottlenecks",
]
NEW_MARKDOWN = [
    Path("README.md"),
    Path("educational_images/README.md"),
    Path("educational_images/v3_CREATION_NOTES.md"),
    Path("educational_images/v3_visual_design_spec.md"),
    Path("educational_images/v3_visual_quality_audit.md"),
    Path("educational_images/v3_visual_before_after_review.md"),
    Path("experiments/v3_detector_integration/README.md"),
    Path("reports/v3_event_detector_evidence.md"),
    Path("reports/v3_full_integration_results.md"),
    Path("reports/v3_limitations_and_negative_results.md"),
    Path("reports/v3_reproducibility.md"),
    Path("reports/v3_third_party_license_audit.md"),
]
PROHIBITED_SUFFIXES = {
    ".edf",
    ".bdf",
    ".onnx",
    ".pt",
    ".pth",
    ".ckpt",
    ".npy",
    ".npz",
    ".h5",
    ".hdf5",
    ".feather",
    ".parquet",
}
PRIVATE_PATTERNS = {
    "windows_user_path": re.compile(r"[A-Za-z]:\\Users\\", re.IGNORECASE),
    "codex_worktree_path": re.compile(r"\.codex[\\/]worktrees", re.IGNORECASE),
    "private_key": re.compile(r"BEGIN [A-Z ]*PRIVATE KEY"),
    "openai_secret": re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
}
LINK_RE = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")


def run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command, cwd=ROOT, text=True, capture_output=True, check=False
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def png_dimensions(path: Path) -> tuple[int, int]:
    with path.open("rb") as handle:
        signature = handle.read(24)
    if signature[:8] != b"\x89PNG\r\n\x1a\n" or signature[12:16] != b"IHDR":
        raise ValueError(f"not a valid PNG: {path}")
    return struct.unpack(">II", signature[16:24])


def changed_paths() -> list[Path]:
    values: set[str] = set()
    for command in (
        ["git", "diff", "--name-only"],
        ["git", "diff", "--cached", "--name-only"],
        ["git", "ls-files", "--others", "--exclude-standard"],
    ):
        result = run(command)
        if result.returncode:
            raise RuntimeError(result.stderr.strip())
        values.update(
            line.strip() for line in result.stdout.splitlines() if line.strip()
        )
    return [Path(value) for value in sorted(values)]


def public_paths() -> list[Path]:
    values: set[str] = set()
    for command in (
        ["git", "ls-files"],
        ["git", "ls-files", "--others", "--exclude-standard"],
    ):
        result = run(command)
        if result.returncode:
            raise RuntimeError(result.stderr.strip())
        values.update(
            line.strip() for line in result.stdout.splitlines() if line.strip()
        )
    values.discard("PUBLIC_MANIFEST.json")
    return [Path(value) for value in sorted(values) if (ROOT / value).is_file()]


def check_links(path: Path) -> list[str]:
    errors: list[str] = []
    text = (ROOT / path).read_text(encoding="utf-8")
    for raw_target in LINK_RE.findall(text):
        target = raw_target.strip().strip("<>").split("#", 1)[0]
        if not target or target.startswith(("http://", "https://", "mailto:", "#")):
            continue
        resolved = (ROOT / path.parent / target).resolve()
        if not resolved.exists():
            errors.append(f"{path.as_posix()} -> {target}")
    return errors


def add(checks: list[dict[str, Any]], name: str, passed: bool, detail: Any) -> None:
    checks.append(
        {"name": name, "status": "PASS" if passed else "FAIL", "detail": detail}
    )


def validate() -> tuple[dict[str, Any], dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    branch = run(["git", "branch", "--show-current"]).stdout.strip()
    add(checks, "exact_target_branch", branch == TARGET_BRANCH, branch)
    ancestor = run(["git", "merge-base", "--is-ancestor", BASE_COMMIT, "HEAD"])
    add(checks, "base_commit_is_ancestor", ancestor.returncode == 0, BASE_COMMIT)
    core = run(["git", "diff", "--quiet", BASE_COMMIT, "--", "src", "configs"])
    add(
        checks,
        "frozen_core_and_configs_unchanged",
        core.returncode == 0,
        core.returncode,
    )

    evidence_path = ROOT / "reports" / "v3_evidence_summary.json"
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    add(
        checks,
        "sealed_evidence_json_unchanged",
        sha256(evidence_path) == EVIDENCE_SHA256,
        sha256(evidence_path),
    )
    arms = evidence["hmc_integration"]["arms"]
    scientific_assertions = {
        "eight_arms": len(arms) == 8,
        "best_accuracy_b100": max(arms, key=lambda key: arms[key]["accuracy"])
        == "B100",
        "best_macro_f1_b101": max(arms, key=lambda key: arms[key]["macro_f1"])
        == "B101",
        "b111_not_best": arms["B111"]["accuracy"] < arms["B100"]["accuracy"],
        "wake_partition_closes": sum(
            evidence["wake_n3_audit"]["wake"]["primary_failure_causes"].values()
        )
        == 1108,
        "n3_partition_closes": sum(
            evidence["wake_n3_audit"]["n3"]["primary_failure_causes"].values()
        )
        == 1153,
        "sleepfm_negative_preserved": evidence["negative_results"][
            "sleepfm_reliability"
        ]["status"]
        == "NO_ADDED_VALUE_FROM_FROZEN_SLEEPFM",
    }
    add(
        checks,
        "scientific_metric_assertions",
        all(scientific_assertions.values()),
        scientific_assertions,
    )

    image_manifest: dict[str, Any] = {"schema_version": 1, "assets": {}}
    image_errors: list[str] = []
    for stem in STEMS:
        png = ROOT / "educational_images" / f"{stem}.png"
        svg = ROOT / "educational_images" / f"{stem}.svg"
        try:
            width, height = png_dimensions(png)
            svg_text = svg.read_text(encoding="utf-8")
            if "<svg" not in svg_text:
                image_errors.append(f"{stem}: invalid SVG")
            if (width, height) != PNG_SIZE:
                image_errors.append(f"{stem}: PNG is {width}x{height}")
            if width >= height or width / height != 0.75:
                image_errors.append(f"{stem}: PNG is not 3:4 portrait")
            if f'viewBox="{SVG_VIEWBOX}"' not in svg_text:
                image_errors.append(
                    f"{stem}: SVG viewBox does not match portrait contract"
                )
            image_manifest["assets"][stem] = {
                "png": {
                    "sha256": sha256(png),
                    "bytes": png.stat().st_size,
                    "width": width,
                    "height": height,
                },
                "svg": {"sha256": sha256(svg), "bytes": svg.stat().st_size},
            }
        except (FileNotFoundError, ValueError) as exc:
            image_errors.append(str(exc))
    add(
        checks,
        "seven_png_and_svg_pairs",
        not image_errors,
        image_errors or "7/7 valid at 1800x2400 with 3:4 portrait SVG pairs",
    )

    synthetic_labels = {
        stem: "合成教学波形 · 非临床 PSG"
        in (ROOT / "educational_images" / f"{stem}.svg").read_text(encoding="utf-8")
        for stem in STEMS[1:5]
    }
    add(
        checks,
        "synthetic_waveform_labels",
        all(synthetic_labels.values()),
        synthetic_labels,
    )

    figure_tokens = {
        STEMS[0]: [
            "N3  →  Wake  →  REM  →  N2  →  N1",
            "仅 unresolved epochs 进入 Pass 2",
            "UNDETERMINED (U)",
        ],
        STEMS[1]: ["0.5788", "0.6580", "9.9", "51.3", "15.0", "52.67"],
        STEMS[2]: ["0.2944", "0.4604", "0.1356", "0.2433", "209.8", "89.4", "net -47"],
        STEMS[3]: [
            "0.6989",
            "0.5798",
            "0.6338",
            "12.77",
            "3.90",
            "60 scorer-record comparisons",
        ],
        STEMS[4]: [
            "0.4593",
            "0.4750",
            "86.26",
            "64.98",
            "0.4333",
            "0.4179",
            "AntiPhase 已存在于 Enhanced V2",
        ],
        STEMS[5]: ["B000", "B100", "B101", "B111", "0.5388", "0.4316", "net -47"],
        STEMS[6]: [
            "1,176 epochs",
            "68 epochs",
            "5.78%",
            "1,534 epochs",
            "381 epochs",
            "24.84%",
            "N3.1 rule-positive: ≥ 6 s",
            "Stage error  ≠  Event detector false negative",
        ],
    }
    token_results: dict[str, bool] = {}
    for stem, tokens in figure_tokens.items():
        svg_text = (ROOT / "educational_images" / f"{stem}.svg").read_text(
            encoding="utf-8"
        )
        token_results[stem] = all(token in svg_text for token in tokens)
    add(
        checks,
        "rendered_scientific_text_contract",
        all(token_results.values()),
        token_results,
    )

    generator_text = (ROOT / "scripts" / "generate_v3_educational_images.py").read_text(
        encoding="utf-8"
    )
    add(
        checks,
        "portrait_font_and_source_contract",
        all(
            token in generator_text
            for token in (
                "PNG_SIZE = (1800, 2400)",
                'FONT_NAME = "Microsoft YaHei"',
                "data = load_evidence()",
            )
        ),
        "1800x2400; Microsoft YaHei fail-fast; metrics loaded from evidence JSON",
    )

    visual_audit_path = ROOT / "educational_images" / "v3_visual_quality_audit.md"
    visual_audit_text = (
        visual_audit_path.read_text(encoding="utf-8")
        if visual_audit_path.exists()
        else ""
    )
    visual_audit_checks = {
        "overall_pass": "Overall: PASS" in visual_audit_text,
        "all_figures_listed": all(stem in visual_audit_text for stem in STEMS),
        "three_view_contract": all(
            token in visual_audit_text for token in ("1800×2400", "800 px", "局部放大")
        ),
    }
    add(
        checks,
        "human_visual_audit_record",
        all(visual_audit_checks.values()),
        visual_audit_checks,
    )

    link_errors = [error for path in NEW_MARKDOWN for error in check_links(path)]
    add(
        checks,
        "relative_markdown_links",
        not link_errors,
        link_errors or "all local links resolve",
    )

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    first_image = re.search(r"!\[[^\]]*\]\(([^)]+)\)", readme)
    readme_checks = {
        "first_image_is_legacy_flow": bool(
            first_image
            and "v3_01_events_to_stages_pass1_pass2.png" in first_image.group(1)
        ),
        "names_disambiguated": all(
            token in readme
            for token in ("main Legacy V1", "experimental `V3`", "B000–B111")
        ),
        "negative_results_present": all(
            token in readme
            for token in ("NO_ADDED_VALUE_FROM_FROZEN_SLEEPFM", "8,352", "SN011–SN154")
        ),
        "fresh_reproduction_not_overclaimed": "不能声明 `FULL_INTEGRATION_PASS`"
        in readme,
    }
    add(checks, "readme_scientific_story", all(readme_checks.values()), readme_checks)

    paths = changed_paths()
    publication_errors: list[str] = []
    for relative in paths:
        path = ROOT / relative
        if not path.is_file():
            continue
        if path.suffix.lower() in PROHIBITED_SUFFIXES:
            publication_errors.append(f"prohibited suffix: {relative.as_posix()}")
        if path.stat().st_size > 5 * 1024 * 1024:
            publication_errors.append(f"file exceeds 5 MiB: {relative.as_posix()}")
        if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".gif", ".pdf"}:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            publication_errors.append(f"unexpected binary file: {relative.as_posix()}")
            continue
        for name, pattern in PRIVATE_PATTERNS.items():
            if pattern.search(text):
                publication_errors.append(f"{name}: {relative.as_posix()}")
    add(
        checks,
        "publication_privacy_and_size",
        not publication_errors,
        publication_errors or f"{len(paths)} changed files scanned",
    )

    manifest_path = ROOT / "PUBLIC_MANIFEST.json"
    manifest_errors: list[str] = []
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected_paths = public_paths()
        expected_names = {path.as_posix() for path in expected_paths}
        manifest_names = set(manifest["files"])
        if expected_names != manifest_names:
            manifest_errors.append(
                f"file set mismatch: missing={sorted(expected_names - manifest_names)}, extra={sorted(manifest_names - expected_names)}"
            )
        for relative in expected_paths:
            name = relative.as_posix()
            entry = manifest["files"].get(name)
            if entry is None:
                continue
            path = ROOT / relative
            if entry["sha256"] != sha256(path) or entry["bytes"] != path.stat().st_size:
                manifest_errors.append(f"hash/size mismatch: {name}")
        if manifest.get("branch") != TARGET_BRANCH:
            manifest_errors.append("manifest branch mismatch")
        if manifest.get("base_commit") != BASE_COMMIT:
            manifest_errors.append("manifest base commit mismatch")
        if manifest.get("frozen_core_or_configs_modified") is not False:
            manifest_errors.append("manifest reports frozen core/config changes")
    except (FileNotFoundError, KeyError, json.JSONDecodeError) as exc:
        manifest_errors.append(str(exc))
    add(
        checks,
        "public_manifest_integrity",
        not manifest_errors,
        manifest_errors or "file set and hashes match",
    )

    tests = run(
        [sys.executable, "-m", "pytest", "tests/test_v3_detector_integration.py", "-q"]
    )
    add(
        checks,
        "v3_contract_tests",
        tests.returncode == 0,
        (tests.stdout + tests.stderr).strip(),
    )

    failed = [check["name"] for check in checks if check["status"] != "PASS"]
    report = {
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if not failed else "FAIL",
        "branch": branch,
        "base_commit": BASE_COMMIT,
        "checks": checks,
        "failed_checks": failed,
        "changed_files_scanned": [path.as_posix() for path in paths],
    }
    return report, image_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-report", action="store_true")
    args = parser.parse_args()
    report, image_manifest = validate()
    if args.write_report:
        report_path = ROOT / "reports" / "v3_release_validation.json"
        image_path = ROOT / "educational_images" / "v3_asset_manifest.json"
        with report_path.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(report, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        with image_path.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(image_manifest, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
    print(
        json.dumps(
            {"status": report["status"], "failed_checks": report["failed_checks"]},
            indent=2,
        )
    )
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
