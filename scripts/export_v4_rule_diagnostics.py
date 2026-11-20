"""Export aggregate-only V4 diagnostics from a frozen rule-discovery run."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import shutil
import sqlite3
import subprocess
import zlib
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


STAGES = ("W", "N1", "N2", "N3", "R", "UNDETERMINED")
STAGE_TO_CODE = {stage: index for index, stage in enumerate(STAGES)}
CONFIGS = ("C000", "C196")
PARTITIONS = ("full", "discovery", "confirmation")
EXPECTED_PRIMARY_EPOCHS = {"full": 8902, "discovery": 4540, "confirmation": 4362}
EXPECTED_CORRECT = {
    "C000": {"full": 4947, "discovery": 2359, "confirmation": 2588},
    "C196": {"full": 5044, "discovery": 2398, "confirmation": 2646},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export frozen M6408 C000/C196 aggregate rule diagnostics."
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/release/v4/rule_diagnostics"),
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path: Path, fields: list[str], rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=fields, extrasaction="raise", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True) + "\n")


def load_reference(path: Path) -> dict[str, list[dict[str, Any]]]:
    records: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with gzip.open(path, "rt", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            records[row["record_id"]].append(
                {
                    "epoch_index": int(row["epoch_index"]),
                    "truth": STAGE_TO_CODE.get(row["true_stage"], 255),
                    "primary": row["in_primary"].strip().lower()
                    in {"1", "true", "yes"},
                }
            )
    for rows in records.values():
        rows.sort(key=lambda row: row["epoch_index"])
        if [row["epoch_index"] for row in rows] != list(range(len(rows))):
            raise RuntimeError("reference epoch indices are not contiguous")
    return dict(records)


def open_cache(path: Path) -> sqlite3.Connection:
    uri = f"file:///{path.resolve().as_posix()}?mode=ro&immutable=1"
    connection = sqlite3.connect(uri, uri=True)
    connection.execute("PRAGMA query_only=ON")
    return connection


def cache_payload(
    connection: sqlite3.Connection, config_id: str, record_id: str
) -> tuple[list[int], list[str], str]:
    row = connection.execute(
        """SELECT stage_codes,rule_ids_zlib,status,epoch_count,output_sha256
           FROM config_records WHERE config_id=? AND record_id=?""",
        (config_id, record_id),
    ).fetchone()
    if row is None or row[2] != "PASS":
        raise RuntimeError(f"missing sealed cache entry: {config_id}")
    predictions = list(row[0])
    rules = json.loads(zlib.decompress(row[1]).decode("utf-8"))
    if len(predictions) != int(row[3]) or len(rules) != int(row[3]):
        raise RuntimeError(f"invalid cache payload length: {config_id}")
    return predictions, rules, str(row[4])


def config_identity(connection: sqlite3.Connection, config_id: str) -> dict[str, Any]:
    row = connection.execute(
        "SELECT config_json,config_sha256 FROM configs WHERE config_id=?", (config_id,)
    ).fetchone()
    if row is None:
        raise RuntimeError(f"missing config identity: {config_id}")
    return {"config": json.loads(row[0]), "config_sha256": row[1]}


def collect_partition(
    connection: sqlite3.Connection,
    config_id: str,
    reference: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    by_rule: dict[str, Counter[str]] = defaultdict(Counter)
    by_rule_stage: Counter[tuple[str, str, str]] = Counter()
    confusion: Counter[tuple[str, str]] = Counter()
    output_hashes: list[str] = []
    for record_id in sorted(reference):
        rows = reference[record_id]
        predictions, rules, output_sha256 = cache_payload(
            connection, config_id, record_id
        )
        output_hashes.append(output_sha256)
        if len(rows) != len(predictions):
            raise RuntimeError(f"prediction/reference length mismatch: {config_id}")
        for row, prediction, rule_id in zip(rows, predictions, rules):
            if not row["primary"]:
                continue
            truth = int(row["truth"])
            predicted_stage = STAGES[int(prediction)]
            reference_stage = STAGES[truth]
            values = by_rule[str(rule_id)]
            values["count"] += 1
            values["correct"] += int(int(prediction) == truth)
            by_rule_stage[(str(rule_id), predicted_stage, reference_stage)] += 1
            confusion[(reference_stage, predicted_stage)] += 1
    return {
        "record_count": len(reference),
        "epoch_count": sum(values["count"] for values in by_rule.values()),
        "correct": sum(values["correct"] for values in by_rule.values()),
        "by_rule": by_rule,
        "by_rule_stage": by_rule_stage,
        "confusion": confusion,
        "cache_output_sha256s": sorted(output_hashes),
    }


def verify_original_diagnostics(
    source_path: Path, computed: dict[str, Counter[str]]
) -> None:
    with source_path.open("r", encoding="utf-8", newline="") as stream:
        source_map = {row["rule_id"]: row for row in csv.DictReader(stream)}
    if set(source_map) != set(computed):
        raise RuntimeError("original current_rule_diagnostics rule set mismatch")
    for rule_id, values in computed.items():
        row = source_map[rule_id]
        if (
            int(row["count"]) != values["count"]
            or int(row["correct"]) != values["correct"]
        ):
            raise RuntimeError(f"original current_rule_diagnostics mismatch: {rule_id}")


def success(value: int, total: int) -> str:
    return "" if total == 0 else f"{value / total:.12f}"


def pct(value: int, total: int) -> str:
    return "NA" if total == 0 else f"{100 * value / total:.2f}%"


def wake_path_category(rule_id: str) -> str:
    if rule_id in {"W.1", "W.2", "W.3"}:
        return "DIRECT_WAKE_RULE"
    if rule_id.startswith("FALLBACK"):
        return "FALLBACK"
    if rule_id.startswith(("MBM", "MOVEMENT")):
        return "MOVEMENT"
    return "OTHER_RULE_PATH"


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    lines.extend("| " + " | ".join(str(value) for value in row) + " |" for row in rows)
    return "\n".join(lines)


def build_readme(
    analyses: dict[tuple[str, str], dict[str, Any]], evidence: dict[str, Any]
) -> str:
    wake_rows: list[list[Any]] = []
    direct_rows: list[list[Any]] = []
    all_rule_rows: list[list[Any]] = []
    for config_id in CONFIGS:
        values = analyses[(config_id, "full")]["by_rule"]
        for rule_id in ("W.1", "W.2", "W.3"):
            row = values.get(rule_id, Counter())
            count = int(row["count"])
            correct = int(row["correct"])
            direct_rows.append(
                [
                    config_id,
                    rule_id,
                    count,
                    correct,
                    count - correct,
                    pct(correct, count),
                    "NO_FIRINGS" if count == 0 else "OBSERVED",
                ]
            )
        wake_paths: Counter[tuple[str, str]] = Counter()
        for (rule_id, predicted, reference), count in analyses[(config_id, "full")][
            "by_rule_stage"
        ].items():
            if predicted == "W":
                category = wake_path_category(rule_id)
                wake_paths[(category, rule_id)] += count
                wake_paths[(category, "__CORRECT__")] += count * int(reference == "W")
        for category in (
            "DIRECT_WAKE_RULE",
            "FALLBACK",
            "MOVEMENT",
            "OTHER_RULE_PATH",
        ):
            rules = sorted(
                rule
                for cat, rule in wake_paths
                if cat == category and rule != "__CORRECT__"
            )
            count = sum(wake_paths[(category, rule)] for rule in rules)
            correct = wake_paths[(category, "__CORRECT__")]
            wake_rows.append(
                [
                    config_id,
                    category,
                    ", ".join(rules) or "NONE",
                    count,
                    correct,
                    count - correct,
                    pct(correct, count),
                ]
            )
        for rule_id in sorted(values, key=lambda key: (-values[key]["count"], key)):
            row = values[rule_id]
            count = int(row["count"])
            correct = int(row["correct"])
            all_rule_rows.append(
                [
                    config_id,
                    rule_id,
                    count,
                    correct,
                    count - correct,
                    pct(correct, count),
                ]
            )
        if "W.2" not in values:
            all_rule_rows.append([config_id, "W.2", 0, 0, 0, "NA"])

    evidence_counts = evidence.get("counts", {})
    evidence_rows = [
        ["alpha > 15 s", evidence_counts.get("alpha_over_15", "NA")],
        ["blink observed", evidence_counts.get("blink_observed", "NA")],
        ["blink > 15 s", evidence_counts.get("blink_over_15", "NA")],
        [
            "irregular REM + normal tone > 15 s",
            evidence_counts.get("irregular_rem_normal_over_15", "NA"),
        ],
        ["LAMF > 15 s", evidence_counts.get("lamf_over_15", "NA")],
        [
            "reference N3: no slow-wave evidence",
            evidence_counts.get("n3_no_slow_wave", "NA"),
        ],
        [
            "reference N3: slow-wave < 6 s",
            evidence_counts.get("n3_slow_wave_below_6", "NA"),
        ],
        [
            "reference N3: slow-wave >= 6 s",
            evidence_counts.get("n3_slow_wave_at_least_6", "NA"),
        ],
    ]
    delta_correct = (
        analyses[("C196", "full")]["correct"] - analyses[("C000", "full")]["correct"]
    )
    return f"""# V4 aggregate rule diagnostics

本目录发布固定 `M6408` 感知输入下的聚合规则诊断。`C000` 是原始 Temporal R1，`C196` 是冻结的两模块规则候选；所有主表均来自同一十晚开发范围的 sealed cache，未重新运行 detector、规则搜索或阈值优化。公开文件不含逐 epoch、逐夜、患者、标签文件、Event Bank 或 Rule Trace。

## W.1 / W.2 / W.3 最终命中

口径是规则成为该 epoch **最终决策规则**，而不是条件曾经满足。`success_rate = correct / count`；零命中显示 `NO_FIRINGS`，成功率为 `NA`，不解释成 0%。

{markdown_table(["Rule profile", "Rule", "Count", "Correct", "Wrong", "Success", "Status"], direct_rows)}

`W.2` 为零的直接原因是冻结 evidence 中 blink 最长只有 {evidence.get('maxima_seconds', {}).get('blink_seconds', 'NA')} 秒，`blink > 15 s` 可达次数为 {evidence_counts.get('blink_over_15', 'NA')}；这说明当前 evidence/rule 组合不可达，不表示专家确认没有 blink。

## Wake 输出来自哪些最终路径

以下只统计 `predicted_stage = W`。Direct 是 W.1/W.2/W.3；Fallback 以实际 `FALLBACK.*` rule ID 计；Movement 仅统计最终 rule ID 明确为 MBM/MOVEMENT 的 Wake 输出。

{markdown_table(["Rule profile", "Path class", "Final rule IDs", "Wake outputs", "Correct", "Wrong", "Success"], wake_rows)}

因此，C000 的 82 个 Wake 输出中 73 个来自 W.1/W.3，另外 9 个来自 `FALLBACK.INHERIT_PREVIOUS`；没有 Movement rule 直接输出 Wake。全 Wake precision 80/82 不能替代 W.1、W.2、W.3 各自的成功率。

同一 primary mask 中真实 Wake support 为 {evidence_counts.get('reference_W', 'NA')}；C000/C196 均只输出 82 个 Wake，其中 80 个正确，即全 Wake precision 97.56%、recall 6.80%。这说明直接 Wake 规则很精确，但覆盖严重不足。

## 全部最终规则命中摘要（full / primary）

{markdown_table(["Rule profile", "Final rule ID", "Count", "Correct", "Wrong", "Success"], all_rule_rows)}

## Evidence 可达性摘要

原始 `evidence_rule_audit.json` 被逐字复制；其函数把 `seconds is None` 转为 0 秒。因此 UNKNOWN/缺失时长会进入零秒侧，不能解释为专家确认事件不存在。

{markdown_table(["Frozen evidence aggregate", "Epoch count"], evidence_rows)}

## C000 与 C196

C196保持 M6408、Event Banks、Evidence、QC 和其他 Temporal R1 逻辑不变，只加入 `R.BACKWARD_BRIDGE` 与 `N2_POST_AROUSAL_REENTRY`。在 full/primary 的 8,902 个 epoch 上，C000 正确 4,947，C196 正确 5,044，净变化 {delta_correct:+d}。这些是同一反复使用开发集的聚合阶段结果，不是外部验证，也不能反推出 Event Detector 准确率。

## 文件

- `current_rule_diagnostics.csv`：原始 C000 聚合表的字段与数值保留副本（仅规范化换行；M6408 + Temporal R1，full/primary）。
- `evidence_rule_audit.json`：原始 evidence reachability 聚合的逐字副本。
- `final_rule_diagnostics.csv`：C000/C196 × full/discovery/confirmation 的最终规则命中、正确、错误和成功率。
- `rule_stage_counts.csv`：最终 rule ID × predicted stage × reference stage 聚合；可重建混淆矩阵。
- `provenance.json`：输入身份、哈希、mask、分区规模、导出方式和一致性检查。

暂不发布逐 epoch、逐夜、记录名单、完整 Trace 或缓存。后续分析 N1/N2/N3/REM 应直接复用 `final_rule_diagnostics.csv` 和 `rule_stage_counts.csv` 的同一口径。

受控环境中的聚合复现命令为：

```powershell
python scripts/export_v4_rule_diagnostics.py `
  --run-dir <sealed-rule-discovery-run> `
  --output-dir results/release/v4/rule_diagnostics
```
"""


def git_commit(repo: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        text=True,
        capture_output=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else "UNAVAILABLE"


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    output_dir = args.output_dir.resolve()
    source_current = run_dir / "analysis" / "current_rule_diagnostics.csv"
    source_evidence = run_dir / "analysis" / "evidence_rule_audit.json"
    cache_path = run_dir / "cache" / "rule_predictions.sqlite"
    source_config = run_dir / "selection" / "final_rule_config.json"
    implementation_manifest = run_dir / "implementation_manifest.json"
    heritage_manifest = run_dir / "heritage_manifest.json"
    required = [source_current, source_evidence, cache_path, source_config]
    required.extend(run_dir / "reference" / f"{name}.csv.gz" for name in PARTITIONS)
    missing = [path.name for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing frozen source artifacts: {missing}")

    cache_hash_before = sha256_file(cache_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    connection = open_cache(cache_path)
    references = {
        name: load_reference(run_dir / "reference" / f"{name}.csv.gz")
        for name in PARTITIONS
    }
    analyses = {
        (config_id, partition): collect_partition(
            connection, config_id, references[partition]
        )
        for config_id in CONFIGS
        for partition in PARTITIONS
    }

    for partition, expected in EXPECTED_PRIMARY_EPOCHS.items():
        for config_id in CONFIGS:
            analysis = analyses[(config_id, partition)]
            if analysis["epoch_count"] != expected:
                raise RuntimeError(
                    f"primary epoch mismatch: {config_id}/{partition}="
                    f"{analysis['epoch_count']}"
                )
            if sum(analysis["by_rule_stage"].values()) != expected:
                raise RuntimeError(
                    f"rule-stage count mismatch: {config_id}/{partition}"
                )
            diagonal = sum(
                count
                for (reference, prediction), count in analysis["confusion"].items()
                if reference == prediction and reference != "UNDETERMINED"
            )
            if diagonal != analysis["correct"]:
                raise RuntimeError(f"confusion mismatch: {config_id}/{partition}")
    for config_id, partitions in EXPECTED_CORRECT.items():
        for partition, expected in partitions.items():
            observed = analyses[(config_id, partition)]["correct"]
            if observed != expected:
                raise RuntimeError(
                    f"correct mismatch: {config_id}/{partition}={observed}"
                )

    verify_original_diagnostics(source_current, analyses[("C000", "full")]["by_rule"])
    c000_wake = sum(
        count
        for (_, predicted, _), count in analyses[("C000", "full")][
            "by_rule_stage"
        ].items()
        if predicted == "W"
    )
    c000_wake_correct = sum(
        count
        for (_, predicted, reference), count in analyses[("C000", "full")][
            "by_rule_stage"
        ].items()
        if predicted == reference == "W"
    )
    if (c000_wake, c000_wake_correct) != (82, 80):
        raise RuntimeError(f"C000 Wake check failed: {(c000_wake, c000_wake_correct)}")

    with (output_dir / "current_rule_diagnostics.csv").open(
        "w", encoding="utf-8", newline="\n"
    ) as stream:
        stream.write(source_current.read_text(encoding="utf-8").replace("\r\n", "\n"))
    shutil.copyfile(source_evidence, output_dir / "evidence_rule_audit.json")

    profile_rules = {
        config_id: set(analyses[(config_id, "full")]["by_rule"]) | {"W.1", "W.2", "W.3"}
        for config_id in CONFIGS
    }
    diagnostics_rows: list[dict[str, Any]] = []
    stage_rows: list[dict[str, Any]] = []
    for config_id in CONFIGS:
        for partition in PARTITIONS:
            analysis = analyses[(config_id, partition)]
            for rule_id in sorted(profile_rules[config_id]):
                values = analysis["by_rule"].get(rule_id, Counter())
                count = int(values["count"])
                correct = int(values["correct"])
                diagnostics_rows.append(
                    {
                        "detector_profile": "M6408",
                        "rule_profile": config_id,
                        "partition": partition,
                        "window": "primary",
                        "rule_id": rule_id,
                        "count": count,
                        "correct": correct,
                        "wrong": count - correct,
                        "success_rate": success(correct, count),
                        "status": "OBSERVED" if count else "NO_FIRINGS",
                    }
                )
            for (rule_id, predicted, reference), count in sorted(
                analysis["by_rule_stage"].items()
            ):
                stage_rows.append(
                    {
                        "detector_profile": "M6408",
                        "rule_profile": config_id,
                        "partition": partition,
                        "window": "primary",
                        "final_rule_id": rule_id,
                        "predicted_stage": predicted,
                        "reference_stage": reference,
                        "count": count,
                    }
                )

    diagnostics_path = output_dir / "final_rule_diagnostics.csv"
    stage_path = output_dir / "rule_stage_counts.csv"
    write_csv(
        diagnostics_path,
        [
            "detector_profile",
            "rule_profile",
            "partition",
            "window",
            "rule_id",
            "count",
            "correct",
            "wrong",
            "success_rate",
            "status",
        ],
        diagnostics_rows,
    )
    write_csv(
        stage_path,
        [
            "detector_profile",
            "rule_profile",
            "partition",
            "window",
            "final_rule_id",
            "predicted_stage",
            "reference_stage",
            "count",
        ],
        stage_rows,
    )

    evidence = json.loads(source_evidence.read_text(encoding="utf-8"))
    with (output_dir / "README.md").open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(build_readme(analyses, evidence))

    repo = Path(__file__).resolve().parents[1]
    source_config_value = json.loads(source_config.read_text(encoding="utf-8"))
    partition_manifest: dict[str, Any] = {}
    for partition in PARTITIONS:
        reference_path = run_dir / "reference" / f"{partition}.csv.gz"
        partition_manifest[partition] = {
            "record_count": len(references[partition]),
            "reference_row_count": sum(
                len(rows) for rows in references[partition].values()
            ),
            "primary_epoch_count": analyses[("C000", partition)]["epoch_count"],
            "mask_definition": "in_primary == true in the frozen reference shard",
            "reference_artifact": reference_path.name,
            "reference_sha256": sha256_file(reference_path),
        }
    validations = {
        "rule_count_sum_matches_mask": True,
        "correct_plus_wrong_matches_count": True,
        "rule_stage_counts_rebuild_confusion": True,
        "original_c000_diagnostics_exactly_recomputed": True,
        "full_primary_epoch_count": 8902,
        "full_correct": {
            config_id: analyses[(config_id, "full")]["correct"] for config_id in CONFIGS
        },
        "correct_by_partition": {
            config_id: {
                partition: analyses[(config_id, partition)]["correct"]
                for partition in PARTITIONS
            }
            for config_id in CONFIGS
        },
        "c000_wake_outputs": c000_wake,
        "c000_wake_correct": c000_wake_correct,
    }
    exported = {
        name: sha256_file(output_dir / name)
        for name in (
            "current_rule_diagnostics.csv",
            "evidence_rule_audit.json",
            "final_rule_diagnostics.csv",
            "rule_stage_counts.csv",
        )
    }
    run_git_head = "UNAVAILABLE"
    if heritage_manifest.is_file():
        heritage = json.loads(heritage_manifest.read_text(encoding="utf-8"))
        run_git_head = str(heritage.get("git_head", "UNAVAILABLE"))
    provenance = {
        "schema_version": "1.0",
        "export_source_commit": git_commit(repo),
        "exporter": {
            "artifact": Path(__file__).name,
            "sha256": sha256_file(Path(__file__).resolve()),
        },
        "frozen_run_source_commit": run_git_head,
        "frozen_run_implementation_manifest_sha256": (
            sha256_file(implementation_manifest)
            if implementation_manifest.is_file()
            else None
        ),
        "detector_profile": {
            "id": "M6408",
            "handlers": ["E04", "E11", "SP", "AR"],
        },
        "rule_profiles": {
            config_id: config_identity(connection, config_id) for config_id in CONFIGS
        },
        "selected_rule_config_identity": {
            "config_id": source_config_value.get("config_id"),
            "config_sha256": source_config_value.get("config_sha256"),
            "changed_modules": source_config_value.get("changed_modules", []),
            "scientific_scope": source_config_value.get("scientific_scope"),
        },
        "sealed_cache": {
            "artifact": cache_path.name,
            "sha256": cache_hash_before,
            "configs_exported": list(CONFIGS),
            "record_level_payload_published": False,
        },
        "source_aggregate_artifacts": {
            source_current.name: sha256_file(source_current),
            source_evidence.name: sha256_file(source_evidence),
        },
        "partitions": partition_manifest,
        "export_method": (
            "Read-only immutable SQLite aggregation plus frozen reference-shard "
            "masks; no replay or tuning."
        ),
        "exported_file_sha256": exported,
        "validations": validations,
        "missing_items": (
            ["Original run git commit was not captured."]
            if run_git_head == "UNAVAILABLE"
            else []
        ),
        "privacy_scope": (
            "Aggregate counts only; no record list, local path, epoch row, patient "
            "data, labels, cache, or trace published."
        ),
    }
    write_json(output_dir / "provenance.json", provenance)
    connection.close()
    if sha256_file(cache_path) != cache_hash_before:
        raise RuntimeError("sealed cache hash changed during read-only export")
    print(
        json.dumps(
            {"status": "PASS", "output_dir": str(output_dir), **validations},
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
