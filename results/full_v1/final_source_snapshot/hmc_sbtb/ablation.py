"""A/B/C recomputation on identical sealed physiological features."""
import json
from pathlib import Path
import pandas as pd
import yaml
from .rules import run_rules, UNKNOWN
from .pipeline import prediction_frame
from .io_utils import sha256, csv_write, write_json, atomic_text
from .evaluation import summarize


def ablate(run):
    manifest = json.loads((run / "frozen_run_manifest.json").read_text())
    cfg = yaml.safe_load((run / "run_config.yaml").read_text())
    aligned = pd.read_csv(run / "aligned_evaluation.csv")
    preserved = ["record_id", "epoch_index", "true_stage", "in_primary", "in_full_record"]
    variants = {}
    for variant in ["A", "B", "C"]:
        frames = []
        for record in manifest["successfully_predicted"]:
            folder = run / "records" / record
            checkpoint = json.loads((folder / "checkpoint.json").read_text())
            if sha256(folder / "features.json") != checkpoint["output_hashes"]["features.json"]:
                raise RuntimeError("frozen feature cache changed")
            features = json.loads((folder / "features.json").read_text())
            predictions, traces = run_rules(features, cfg, variant)
            out = run / "ablations" / variant / "records" / record
            out.mkdir(parents=True, exist_ok=True)
            frame = prediction_frame(predictions)
            csv_write(frame, out / "epoch_predictions.csv")
            atomic_text(out / "rule_trace.jsonl", "".join(json.dumps(t, sort_keys=True) + "\n" for t in traces))
            frames.append(frame)
        prediction = pd.concat(frames, ignore_index=True)
        if variant == "C":
            main = pd.read_csv(run / "epoch_predictions.csv")
            keys = ["record_id", "epoch_index", "pass1_prediction", "final_prediction", "decision_source", "rule_id", "event_ids"]
            pd.testing.assert_frame_equal(main[keys].reset_index(drop=True), prediction[keys].reset_index(drop=True), check_dtype=False)
        frame = prediction.merge(aligned[preserved], on=["record_id", "epoch_index"], validate="one_to_one")
        variants[variant] = frame
        out = run / "ablations" / variant
        csv_write(prediction, out / "epoch_predictions.csv")
        summarize(frame, out, cfg["evaluation"])
    changes, summaries = [], []
    for before, after in [("A", "B"), ("B", "C")]:
        old, new = variants[before], variants[after]
        joined = old.merge(new, on=["record_id", "epoch_index"], suffixes=("_old", "_new"), validate="one_to_one")
        changed = joined.final_prediction_old != joined.final_prediction_new
        for _, row in joined[changed].iterrows():
            changes.append({"comparison": before + "→" + after, "record_id": row.record_id,
                            "epoch_index": row.epoch_index, "true_stage": row.true_stage_new,
                            "old_prediction": row.final_prediction_old, "new_prediction": row.final_prediction_new,
                            "new_source": row.decision_source_new, "parent_epoch": row.parent_epoch_new,
                            "anchor_epoch": row.anchor_epoch_new, "chain_length": row.chain_length_new,
                            "in_primary": row.in_primary_new})
        for window in ["primary", "full_record"]:
            rows = joined[joined["in_" + window + "_new"]]
            old_unknown = rows.final_prediction_old == UNKNOWN
            new_known = rows.final_prediction_new != UNKNOWN
            new_correct = rows.final_prediction_new == rows.true_stage_new
            old_correct = rows.final_prediction_old == rows.true_stage_old
            om = json.loads((run / "ablations" / before / "overall_metrics.json").read_text())[window]
            nm = json.loads((run / "ablations" / after / "overall_metrics.json").read_text())[window]
            summaries.append({"comparison": before + "→" + after, "window": window,
                              "newly_correct_from_unknown": int((old_unknown & new_correct).sum()),
                              "newly_wrong_from_unknown": int((old_unknown & new_known & ~new_correct).sum()),
                              "corrected_existing_error": int((~old_unknown & ~old_correct & new_correct).sum()),
                              "harmed_existing_correct": int((old_correct & ~new_correct).sum()),
                              "remaining_undetermined": int((~new_known).sum()),
                              **{key + "_delta": nm[key] - om[key] if nm[key] is not None and om[key] is not None else None
                                 for key in ["accuracy", "macro_f1", "kappa", "prediction_coverage"]}})
    csv_write(pd.DataFrame(changes), run / "prediction_changes.csv")
    write_json(run / "ablation_incremental.json", summaries)
    write_json(run / "ablations" / "manifest.json", {"status": "TESTED", "same_features": True,
        "C_matches_main": True, "feature_hashes": {r: sha256(run / "records" / r / "features.json")
                                                   for r in manifest["successfully_predicted"]}})
    print(json.dumps(summaries, indent=2), flush=True)
    return variants
