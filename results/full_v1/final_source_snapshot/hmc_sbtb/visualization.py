"""Frozen-output physiological audit, actual waveform figures and rule statistics."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from .io_utils import csv_write, write_json
from .temporal import union

COLORS = {"slow_wave": "#427cba", "spindle": "#7f52ab", "alpha": "#47855c",
          "lamf": "#555555", "kcomplex": "#ce883e", "rem_candidate": "#bc4a4a",
          "low_emg": "#428b79", "high_emg": "#cc7d47", "arousal_candidate": "#9d493a"}


def waveform_case(run, row, number, events, features, info, data_root):
    record, epoch = row.record_id, int(row.epoch_index)
    start, end = max(0, epoch * 30 - 15), min(info["duration_sec"], (epoch + 1) * 30 + 15)
    roles = ["frontal", "central", "occipital", "eog_left", "eog_right", "emg"]
    channels = [info["selected_channels"][r] for r in roles]
    available = [c for c in channels if c]
    raw = mne.io.read_raw_edf(data_root / "recordings" / f"{record}.edf", include=available, preload=False, verbose=False)
    sf = raw.info["sfreq"]
    data = raw.get_data(start=round(start * sf), stop=round(end * sf)) * 1e6
    t = np.arange(data.shape[1]) / sf + start
    fig, axes = plt.subplots(6, 1, figsize=(14, 11), sharex=True)
    for ax, role, channel in zip(axes, roles, channels):
        if channel is None:
            ax.text(.4, .5, "UNAVAILABLE", transform=ax.transAxes)
        else:
            index = raw.ch_names.index(channel)
            ax.plot(t[::2], data[index, ::2], color="#252525", lw=.45)
            relevant = events[(events.offset > start) & (events.onset < end)]
            if role.startswith("eog"):
                relevant = relevant[relevant.event_type == "rem_candidate"]
            elif role == "emg":
                relevant = relevant[relevant.event_type.isin(["low_emg", "high_emg"])]
            else:
                relevant = relevant[relevant.channel == channel]
            for event_type, rows in relevant[relevant.qualified].groupby("event_type"):
                first = True
                for a, b in union(list(zip(rows.onset, rows.offset))):
                    ax.axvspan(max(start, a), min(end, b), color=COLORS[event_type], alpha=.18,
                               label=event_type if first else None)
                    first = False
            if len(relevant):
                handles, labels = ax.get_legend_handles_labels()
                if handles:
                    ax.legend(handles, labels, loc="upper right", fontsize=7, ncol=4)
        ax.axvline(epoch * 30, color="#222222", ls="--", lw=.8)
        ax.axvline((epoch + 1) * 30, color="#222222", ls="--", lw=.8)
        ax.set_ylabel((channel or role) + "\n(uV)", fontsize=8)
        ax.grid(alpha=.15)
    f = features[epoch]
    evidence = ", ".join(f"{k}={f[k + '_sec'] if f[k + '_sec'] is not None else 'UNKNOWN'}s"
                         for k in ["slow_wave", "alpha", "lamf", "spindle", "rem_candidate", "low_emg"])
    fig.suptitle(f"{record} epoch {epoch}: human={row.true_stage}; predicted={row.final_prediction}; "
                 f"source={row.decision_source}\nrule={row.rule_id}; parent={row.parent_epoch}; "
                 f"anchor={row.anchor_epoch}; chain={row.chain_length}\n{evidence}", fontsize=10)
    axes[-1].set_xlabel("Time relative to PSG onset (s)")
    fig.tight_layout(rect=[0, 0, 1, .92])
    path = run / "figures" / f"case_{number:02d}_{record}_epoch{epoch}.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    raw.close()
    return path


def audit(run):
    (run / "figures").mkdir(exist_ok=True)
    manifest = json.loads((run / "frozen_run_manifest.json").read_text())
    frame = pd.read_csv(run / "aligned_evaluation.csv")
    primary = frame[frame.in_primary].copy()
    primary["correct"] = primary.true_stage == primary.final_prediction
    statistics = []
    for keys, rows in primary.groupby(["decision_source", "rule_id"]):
        statistics.append({"decision_source": keys[0], "rule_id": keys[1], "epochs": len(rows),
                           "proportion": len(rows) / len(primary), "accuracy": float(rows.correct.mean())})
    csv_write(pd.DataFrame(statistics), run / "rule_statistics.csv")
    primary["chain_bin"] = pd.cut(primary.chain_length, [-1, 0, 5, 20, 100000], labels=["0", "1-5", "6-20", "21+"])
    chains = [{"chain_length_bin": str(k), "epochs": len(g), "error_rate": float((~g.correct).mean())}
              for k, g in primary.groupby("chain_bin", observed=True)]
    csv_write(pd.DataFrame(chains), run / "chain_error_rates.csv")
    detector_rows, quality_rows, background_rows = [], [], []
    for record in manifest["successfully_predicted"]:
        folder = run / "records" / record
        events = pd.read_csv(folder / "events.csv")
        info = json.loads((folder / "signal_info.json").read_text())
        for (kind, qualified), rows in events.groupby(["event_type", "qualified"]):
            detector_rows.append({"record_id": record, "event_type": kind, "qualified": bool(qualified),
                                  "events": len(rows), "events_per_hour": len(rows) / (info["duration_sec"] / 3600),
                                  "median_duration_sec": float(rows.duration.median()),
                                  "rejection_reasons": str(rows.rejection_reason.fillna("").value_counts().to_dict())})
        quality = pd.read_csv(folder / "quality_flags.csv")
        for role, rows in quality.groupby("role"):
            quality_rows.append({"record_id": record, "role": role, "invalid_epochs": int((~rows.valid).sum()),
                                 "total_epochs": len(rows), "line_50hz_warning_epochs": int(rows.line_50hz_warning.sum())})
        backgrounds = json.loads((folder / "background_statistics.json").read_text())
        features = json.loads((folder / "features.json").read_text())
        background_rows.append({"record_id": record, "lamf_threshold_uv": backgrounds["lamf_amplitude_threshold_uv"],
                                "lamf_epoch_count": sum((f["lamf_sec"] or 0) > 0 for f in features),
                                "emg_low_rms_uv": backgrounds.get("emg", {}).get("low_rms_uv"),
                                "emg_high_rms_uv": backgrounds.get("emg", {}).get("high_rms_uv"),
                                "alpha_generator": features[0]["generates_alpha_rhythm"] if features else None})
    csv_write(pd.DataFrame(detector_rows), run / "detector_statistics.csv")
    csv_write(pd.DataFrame(quality_rows), run / "quality_statistics.csv")
    csv_write(pd.DataFrame(background_rows), run / "background_statistics.csv")
    # Two cases per requested error direction where actually present, then deterministic fill.
    chosen, used = [], set()
    for true, pred in [("N1", "N2"), ("N1", "R"), ("W", "N2"), ("W", "R"), ("N3", "N2"), ("R", "N2")]:
        candidates = primary[(primary.true_stage == true) & (primary.final_prediction == pred)]
        for _, row in candidates.drop_duplicates("record_id").head(2).iterrows():
            key = (row.record_id, int(row.epoch_index))
            if key not in used:
                chosen.append(row)
                used.add(key)
    for _, row in primary[~primary.correct].iterrows():
        if len(chosen) >= 10:
            break
        key = (row.record_id, int(row.epoch_index))
        if key not in used:
            chosen.append(row)
            used.add(key)
    for _, row in primary[primary.correct].drop_duplicates(["true_stage"]).head(5).iterrows():
        chosen.append(row)
    cases = []
    for i, row in enumerate(chosen, 1):
        folder = run / "records" / row.record_id
        info = json.loads((folder / "signal_info.json").read_text())
        features = json.loads((folder / "features.json").read_text())
        events = pd.read_csv(folder / "events.csv")
        figure = waveform_case(run, row, i, events, features, info, Path(manifest["data_root"]))
        is_wrong = row.true_stage != row.final_prediction
        f = features[int(row.epoch_index)]
        trace = next(json.loads(line) for line in (folder / "rule_trace.jsonl").open(encoding="utf-8")
                     if json.loads(line)["epoch_index"] == int(row.epoch_index))
        alternatives = {}
        for variant in ["A", "B", "C"]:
            p = run / "ablations" / variant / "records" / row.record_id / "epoch_predictions.csv"
            if p.exists():
                alternative = pd.read_csv(p).iloc[int(row.epoch_index)]
                alternatives[variant] = {"stage": alternative.final_prediction, "source": alternative.decision_source}
        evidence = {k: f[k] for k in f if k.endswith("_sec") or k.endswith("_available")}
        nearby = events[(events.onset >= max(0, row.start_sec - 15)) & (events.onset < row.end_sec)]
        direct_events = nearby[nearby.qualified & nearby.event_type.isin(["spindle", "kcomplex", "slow_wave", "rem_candidate"])]
        atomic_detail = {"case_id": i, "record_id": row.record_id, "epoch_index": int(row.epoch_index),
                         "signal_evidence": evidence, "direct_nearby_events": direct_events.to_dict("records"),
                         "actual_trace": trace, "ablation_predictions": alternatives}
        # pandas may represent rejected/empty fields with NaN; normalize to JSON null.
        atomic_detail["direct_nearby_events"] = json.loads(direct_events.to_json(orient="records"))
        write_json(run / "case_details" / f"case_{i:02d}.json", atomic_detail)
        if row.decision_source == "INHERITANCE":
            verified = f"Previous epoch {row.parent_epoch} supplied {row.final_prediction}; anchor {row.anchor_epoch}; chain {row.chain_length}; A/B/C={alternatives}."
        elif row.decision_source == "CONTINUATION":
            verified = f"{row.rule_id} continued anchor {row.anchor_epoch} through parent {row.parent_epoch}; no new direct event was created. A/B/C={alternatives}."
        else:
            verified = f"Ordered definite check {row.rule_id} won; associated event IDs={row.event_ids}; observed SW={f['slow_wave_sec']}s, alpha={f['alpha_sec']}s, LAMF={f['lamf_sec']}s."
        cases.append({"case_id": i, "record_id": row.record_id, "epoch_index": int(row.epoch_index),
                      "true_stage": row.true_stage, "prediction": row.final_prediction,
                      "decision_source": row.decision_source, "rule_id": row.rule_id,
                      "parent_epoch": row.parent_epoch, "anchor_epoch": row.anchor_epoch, "chain_length": row.chain_length,
                      "error": bool(is_wrong), "figure": str(figure.relative_to(run)),
                      "signal_evidence": json.dumps(evidence, sort_keys=True),
                      "VERIFIED_CAUSE": verified + " This verifies the software decision path, not a clinical detector failure.",
                      "PLAUSIBLE_HYPOTHESIS": "Literal LAMF sensitivity, candidate false positives, or prior-anchor persistence may contribute; not independently event-validated.",
                      "UNRESOLVED": "No independent physiological event labels; a stage mismatch does not identify detector error."})
    csv_write(pd.DataFrame(cases), run / "audit_cases.csv")
    metrics = json.loads((run / "overall_metrics.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, (window, result) in zip(axes, metrics.items()):
        cm = np.array(result["confusion_matrix"])
        normalized = np.divide(cm, cm.sum(axis=1, keepdims=True), out=np.zeros_like(cm, dtype=float), where=cm.sum(axis=1, keepdims=True) > 0)
        ax.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
        for r in range(5):
            for c in range(6):
                ax.text(c, r, f"{cm[r,c]}\n{100 * normalized[r,c]:.1f}%", ha="center", va="center", fontsize=8,
                        color="white" if normalized[r,c] > .6 else "black")
        ax.set_xticks(range(6), result["predicted_labels"], rotation=30, ha="right")
        ax.set_yticks(range(5), result["true_labels"])
        ax.set_xlabel("Predicted stage (counts and row %)")
        ax.set_ylabel("Human stage")
        ax.set_title(f"{window}: Acc={result['accuracy']:.3f}, kappa={result['kappa']:.3f}")
    fig.suptitle(f"HMC {run.name}: frozen predictions; UNDETERMINED retained")
    fig.tight_layout()
    fig.savefig(run / "figures" / "confusion_matrices.png", dpi=150)
    plt.close(fig)
    record = primary.record_id.iloc[0]
    rows = frame[frame.record_id == record]
    lookup = {"W": 0, "N1": 1, "N2": 2, "N3": 3, "R": 4, "UNDETERMINED": 5, "UNALIGNED": 5}
    fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
    for ax, col in zip(axes, ["true_stage", "final_prediction"]):
        ax.step(rows.start_sec / 3600, rows[col].map(lookup), where="post", color="#3b6a91")
        ax.set_yticks(range(6), ["W", "N1", "N2", "N3", "R", "U"])
        ax.set_ylabel(col)
        ax.grid(alpha=.2)
    axes[-1].set_xlabel("Time from PSG onset (hours)")
    fig.suptitle(f"{record}: human vs frozen rule prediction; no label-based state reset")
    fig.tight_layout()
    fig.savefig(run / "figures" / "hypnogram.png", dpi=150)
    plt.close(fig)
    write_json(run / "audit_manifest.json", {"status": "TESTED", "wrong_waveforms": sum(c["error"] for c in cases),
        "correct_waveforms": sum(not c["error"] for c in cases), "event_level_precision_recall": "NOT_EVALUATED: no independent event annotations",
        "attribution_policy": "verified execution path separated from physiological hypotheses"})
    print("AUDIT", len(cases), "real waveform cases", flush=True)
