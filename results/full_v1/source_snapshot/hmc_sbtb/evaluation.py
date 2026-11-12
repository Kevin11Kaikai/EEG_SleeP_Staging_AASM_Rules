"""Evaluation reads sealed predictions and annotations, never calls detectors/rules."""
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from .annotations import edf_annotations, txt_annotations, align
from .io_utils import sha256, write_json, csv_write
from .rules import STAGES, UNKNOWN

JOINT = STAGES + [UNKNOWN]


def metrics(y_true, y_pred):
    cm = np.zeros((5, 6), dtype=np.int64)
    for a, b in zip(y_true, y_pred):
        cm[STAGES.index(a), JOINT.index(b)] += 1
    return metrics_cm(cm)


def metrics_cm(cm):
    cm = np.asarray(cm, dtype=np.int64)
    n = int(cm.sum())
    correct = int(np.trace(cm[:, :5]))
    observed = correct / n if n else None
    row_counts = cm.sum(axis=1)
    col_counts = cm.sum(axis=0)
    expected = float(np.dot(np.r_[row_counts, 0], col_counts)) / (n * n) if n else None
    kappa = (observed - expected) / (1 - expected) if n and expected < 1 else None
    stages = {}
    for i, label in enumerate(STAGES):
        tp, support, predicted = int(cm[i, i]), int(row_counts[i]), int(col_counts[i])
        fp, fn = predicted - tp, support - tp
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
        stages[label] = {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1,
                         "true_count": support, "predicted_count": predicted,
                         "undetermined_count": int(cm[i, 5]), "undetermined_rate": int(cm[i, 5]) / support if support else 0.0}
    return {"epochs": n, "accuracy": observed, "kappa": kappa,
            "kappa_undefined_reason": "no_samples_or_expected_agreement_one" if kappa is None else None,
            "macro_f1": float(np.mean([s["f1"] for s in stages.values()])) if n else None,
            "balanced_accuracy": float(np.mean([s["recall"] for s in stages.values()])) if n else None,
            "undetermined_rate": int(col_counts[5]) / n if n else None,
            "prediction_coverage": 1 - int(col_counts[5]) / n if n else None,
            "wake_prevalence": int(row_counts[0]) / n if n else None,
            "confusion_matrix": cm.tolist(), "true_labels": STAGES, "predicted_labels": JOINT,
            "stage_metrics": stages, "zero_division": 0}


def bootstrap_subjects(frame, cfg):
    subjects = sorted(frame.subject_id.unique())
    cms = np.array([metrics(g.true_stage, g.final_prediction)["confusion_matrix"]
                    for _, g in frame.groupby("subject_id", sort=True)])
    rng = np.random.default_rng(cfg["bootstrap_seed"])
    keys = ["accuracy", "kappa", "macro_f1", "balanced_accuracy"]
    draws, invalid = {k: [] for k in keys}, {k: 0 for k in keys}
    for _ in range(cfg["bootstrap_iterations"]):
        sample = cms[rng.integers(0, len(subjects), len(subjects))].sum(axis=0)
        value = metrics_cm(sample)
        for key in keys:
            if value[key] is None or not math.isfinite(value[key]):
                invalid[key] += 1
            else:
                draws[key].append(value[key])
    return {"unit": "subject_id (record proxy unless confirmed one patient per recording)",
            "subjects": len(subjects), "seed": cfg["bootstrap_seed"], "iterations": cfg["bootstrap_iterations"],
            "invalid_draws": invalid, "invalid_policy": "omit only undefined metric draw, report count",
            "percentile_95": {k: np.quantile(v, [.025, .975]).tolist() if v else None for k, v in draws.items()}}


def summarize(frame, out, cfg, bootstrap=True):
    out.mkdir(parents=True, exist_ok=True)
    all_metrics = {}
    for window in ["primary", "full_record"]:
        selected = frame[frame["in_" + window]]
        result = metrics(selected.true_stage, selected.final_prediction)
        recording, subject = [], []
        for key, rows in selected.groupby("record_id"):
            m = metrics(rows.true_stage, rows.final_prediction)
            recording.append({"window": window, "record_id": key, **{k: m[k] for k in ["epochs", "accuracy", "kappa", "macro_f1", "balanced_accuracy", "undetermined_rate"]}})
        for key, rows in selected.groupby("subject_id"):
            m = metrics(rows.true_stage, rows.final_prediction)
            subject.append({"window": window, "subject_id": key, **{k: m[k] for k in ["epochs", "accuracy", "kappa", "macro_f1", "balanced_accuracy", "undetermined_rate"]}})
        result["subject_macro"] = {key: float(np.mean([r[key] for r in subject if r[key] is not None]))
                                    for key in ["accuracy", "kappa", "macro_f1", "balanced_accuracy"]}
        if bootstrap and len(selected):
            result["bootstrap"] = bootstrap_subjects(selected, cfg)
        result["source_summary"] = [{"decision_source": str(key), "epochs": len(rows),
                                      "proportion": len(rows) / len(selected),
                                      "accuracy": float((rows.true_stage == rows.final_prediction).mean())}
                                     for key, rows in selected.groupby("decision_source")]
        for key in ["accuracy", "kappa"]:
            reference = .605 if key == "accuracy" else .42
            result[key + "_gap"] = (result[key] - reference) * (100 if key == "accuracy" else 1) if result[key] is not None else None
        all_metrics[window] = result
        csv_write(pd.DataFrame(result["confusion_matrix"], columns=JOINT).assign(true_stage=STAGES), out / f"confusion_matrix_{window}.csv")
        csv_write(pd.DataFrame(recording), out / f"recording_metrics_{window}.csv")
        csv_write(pd.DataFrame(subject), out / f"subject_metrics_{window}.csv")
    write_json(out / "overall_metrics.json", all_metrics)
    # Conventional primary-window names.
    for source, target in [("confusion_matrix_primary.csv", "confusion_matrix.csv"),
                            ("recording_metrics_primary.csv", "recording_metrics.csv"),
                            ("subject_metrics_primary.csv", "subject_metrics.csv")]:
        (out / target).write_bytes((out / source).read_bytes())
    return all_metrics


def evaluate(run):
    manifest = json.loads((run / "frozen_run_manifest.json").read_text())
    if manifest["status"] != "PREDICTIONS_FROZEN":
        raise RuntimeError("evaluation requires complete persisted prediction freeze")
    for name, digest in manifest["prediction_output_hashes"].items():
        if sha256(run / name) != digest:
            raise RuntimeError("frozen output tampered: " + name)
    cfg = yaml.safe_load((run / "run_config.yaml").read_text())
    data_root = Path(manifest["data_root"])
    checksums = {line.split(maxsplit=1)[1].strip(): line.split()[0]
                 for line in (data_root / "SHA256SUMS.txt").read_text().splitlines()}
    all_frames, alignment_reports, errors, raw_rows = [], [], [], []
    for record in manifest["successfully_predicted"]:
        try:
            folder = run / "records" / record
            pred = pd.read_csv(folder / "epoch_predictions.csv")
            info = json.loads((folder / "signal_info.json").read_text())
            for suffix in ["_sleepscoring.edf", "_sleepscoring.txt"]:
                relative = f"recordings/{record}{suffix}"
                if sha256(data_root / relative) != checksums[relative]:
                    raise RuntimeError("annotation official checksum mismatch")
            edf = edf_annotations(data_root / "recordings" / f"{record}_sleepscoring.edf", info["start_datetime"])
            txt = txt_annotations(data_root / "recordings" / f"{record}_sleepscoring.txt", info["start_datetime"])
            ae, at = align(edf, len(pred)), align(txt, len(pred))
            rows, disagreements = [], []
            for e, t in zip(ae, at):
                matching = e["true_stage"] == t["true_stage"] and e["true_stage"] in STAGES
                rows.append({**e, "true_stage": e["true_stage"] if matching else "UNALIGNED",
                             "txt_stage": t["true_stage"], "txt_status": t["alignment_status"],
                             "alignment_status": "ALIGNED" if matching else "TXT_EDF_MISMATCH_OR_UNALIGNED"})
                if not matching:
                    disagreements.append({"epoch_index": e["epoch_index"], "edf": e, "txt": t})
            aligned = pd.DataFrame(rows)
            frame = pred.merge(aligned, on="epoch_index", validate="one_to_one")
            frame["in_full_record"] = frame.true_stage.isin(STAGES)
            nonwake = frame[frame.in_full_record & (frame.true_stage != "W")]
            frame["in_primary"] = (frame.in_full_record & (frame.epoch_index >= nonwake.epoch_index.min()) &
                                   (frame.epoch_index <= nonwake.epoch_index.max())) if len(nonwake) else False
            all_frames.append(frame)
            diagnostics = [{"shift_epochs": shift, "matching_epochs": sum(
                ae[i]["true_stage"] == at[i + shift]["true_stage"] and ae[i]["true_stage"] in STAGES
                for i in range(len(ae)) if 0 <= i + shift < len(at))}
                           for shift in range(-2, 3)]
            alignment_reports.append({"record_id": record, "total_complete_epochs": len(pred),
                                      "aligned_epochs": int(frame.in_full_record.sum()),
                                      "agreement_rate": float(frame.in_full_record.mean()),
                                      "trailing_signal_sec": info["duration_sec"] - len(pred) * 30,
                                      "conflict_samples": disagreements[:20], "offset_diagnostics": diagnostics,
                                      "edf_origin_offsets_sec": sorted(set(r["origin_offset_sec"] for r in edf)),
                                      "unknown_raw_labels": sorted(set(r["raw_label"] for r in edf + txt if r["label"] is None)),
                                      "absolute_relative_discrepancies_sec": sorted(set(r["absolute_relative_discrepancy_sec"] for r in txt if r["absolute_relative_discrepancy_sec"] is not None))})
            for source, rows in [("EDF+", edf), ("TXT", txt)]:
                raw_rows.extend({"record_id": record, "source": source, **r} for r in rows)
        except Exception as exc:
            errors.append({"record_id": record, "error": repr(exc)})
            print(record, "EVALUATION_BLOCKED", repr(exc), flush=True)
    write_json(run / "annotation_alignment.json", alignment_reports)
    write_json(run / "evaluation_errors.json", errors)
    csv_write(pd.DataFrame(raw_rows), run / "raw_annotations.csv")
    if not all_frames:
        raise RuntimeError("no records could be evaluated; see evaluation_errors.json")
    frame = pd.concat(all_frames, ignore_index=True)
    csv_write(frame, run / "aligned_evaluation.csv")
    result = summarize(frame, run, cfg["evaluation"])
    # Named pilot and untouched-subject proxy subsets, without changing algorithms.
    for name, records in [("pilot", manifest["pilot_selection"]["pilot"]), ("untouched", manifest["pilot_selection"]["untouched"])]:
        subset = frame[frame.record_id.isin(records)]
        if len(subset):
            summarize(subset, run / "subsets" / name, cfg["evaluation"])
    write_json(run / "evaluation_manifest.json", {"status": "FULL_EVALUATED" if len(all_frames) == manifest["official_count"] else "PILOT_EVALUATED" if len(all_frames) >= 10 else "SMOKE_VERIFIED",
        "records": sorted(frame.record_id.unique()), "failed_records": errors,
        "prediction_sealed_utc": manifest["prediction_artifacts_sealed_utc"],
        "config_sha256": manifest["configuration_sha256"], "prediction_source_sha256": manifest["prediction_source_sha256"],
        "annotation_source_policy": "EDF+ and TXT must agree exactly for full 30-second epoch",
        "predictions_verified": True})
    print(json.dumps({k: {j: v[j] for j in ["epochs", "accuracy", "kappa", "macro_f1"]} for k, v in result.items()}, indent=2), flush=True)
    return result
