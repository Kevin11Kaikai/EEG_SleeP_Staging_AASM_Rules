from __future__ import annotations

from collections import Counter
from typing import Any, Iterable

from .cache import PredictionCache
from .config import STAGES
from .io import load_rows
from .metrics import ReferenceData


def paired_rule_mechanism(
    cache: PredictionCache,
    reference: ReferenceData,
    candidate_id: str,
    records: Iterable[str],
) -> dict[str, Any]:
    """Trace changed outputs without claiming unobserved event correctness."""

    stage_transitions: Counter[str] = Counter()
    rule_transitions: Counter[str] = Counter()
    outcome_types: Counter[str] = Counter()
    reference_stages: Counter[str] = Counter()
    by_record: list[dict[str, Any]] = []
    changed_rows: list[dict[str, Any]] = []
    for record in records:
        data = reference.rows[record]
        mask = data["primary"]
        truth = data["truth"]
        baseline = cache.stage_codes("C000", record)
        candidate = cache.stage_codes(candidate_id, record)
        baseline_rules = cache.rule_ids("C000", record)
        candidate_rules = cache.rule_ids(candidate_id, record)
        local = Counter()
        for index, keep in enumerate(mask.tolist()):
            if not keep or baseline[index] == candidate[index]:
                continue
            y = int(truth[index])
            left = int(baseline[index])
            right = int(candidate[index])
            stage_key = f"{STAGES[left]}->{STAGES[right]}"
            rule_key = f"{baseline_rules[index]}->{candidate_rules[index]}"
            stage_transitions[stage_key] += 1
            rule_transitions[rule_key] += 1
            reference_stages[STAGES[y]] += 1
            if left != y and right == y:
                outcome = "CORRECTED"
            elif left == y and right != y:
                outcome = "INTRODUCED"
            elif left != y and right != y:
                outcome = "CHANGED_BUT_STILL_WRONG"
            else:
                outcome = "NO_ACCURACY_CHANGE"
            outcome_types[outcome] += 1
            local[outcome] += 1
            changed_rows.append(
                {
                    "record_id": record,
                    "epoch_index": index,
                    "reference_stage": STAGES[y],
                    "baseline_prediction": STAGES[left],
                    "candidate_prediction": STAGES[right],
                    "baseline_rule": baseline_rules[index],
                    "candidate_rule": candidate_rules[index],
                    "outcome_type": outcome,
                }
            )
        by_record.append(
            {
                "record_id": record,
                "changed": int(sum(local.values())),
                "corrected": int(local["CORRECTED"]),
                "introduced": int(local["INTRODUCED"]),
                "net_correct": int(local["CORRECTED"] - local["INTRODUCED"]),
                "changed_but_still_wrong": int(local["CHANGED_BUT_STILL_WRONG"]),
            }
        )
    return {
        "candidate_id": candidate_id,
        "records": list(records),
        "stage_transitions": dict(stage_transitions.most_common()),
        "rule_transitions": dict(rule_transitions.most_common()),
        "reference_stage_distribution_among_changes": dict(reference_stages),
        "outcome_types": dict(outcome_types),
        "per_record": by_record,
        "changed_epochs": changed_rows,
        "claim_boundary": (
            "Trace-supported stage mechanism only; HMC has no independent event truth, "
            "so event detector correctness is not inferred."
        ),
    }


def evidence_rule_audit(
    reference: ReferenceData,
    records: Iterable[str],
) -> dict[str, Any]:
    """Quantify reachability and coverage of key frozen rule predicates."""

    counts: Counter[str] = Counter()
    maxima = {"blink_seconds": 0.0, "slow_wave_seconds": 0.0}
    for record in records:
        rows, _ = load_rows(record)
        data = reference.rows[record]
        for index, keep in enumerate(data["primary"].tolist()):
            if not keep:
                continue
            row = rows[index]
            stage = STAGES[int(data["truth"][index])]
            counts["primary_epochs"] += 1
            counts[f"reference_{stage}"] += 1
            counts["invalid_signal"] += int(not row.signals_good)
            blink = float(row.events["blink"].seconds or 0.0)
            alpha = float(row.events["alpha"].seconds or 0.0)
            irregular = float(row.events["irregular_rem_normal"].seconds or 0.0)
            lamf = float(row.events["lamf"].seconds or 0.0)
            slow = float(row.events["slow_wave"].seconds or 0.0)
            maxima["blink_seconds"] = max(maxima["blink_seconds"], blink)
            maxima["slow_wave_seconds"] = max(maxima["slow_wave_seconds"], slow)
            counts["blink_observed"] += int(blink > 0)
            counts["blink_over_15"] += int(blink > 15)
            counts["alpha_over_15"] += int(alpha > 15)
            counts["irregular_rem_normal_over_15"] += int(irregular > 15)
            counts["lamf_over_15"] += int(lamf > 15)
            if stage == "W":
                counts["wake_blink_observed"] += int(blink > 0)
                counts["wake_blink_over_15"] += int(blink > 15)
                counts["wake_alpha_over_15"] += int(alpha > 15)
                counts["wake_irregular_over_15"] += int(irregular > 15)
            if stage == "N1":
                counts["n1_lamf_over_15"] += int(lamf > 15)
            if stage == "N3":
                if slow <= 0:
                    counts["n3_no_slow_wave"] += 1
                elif slow < 6:
                    counts["n3_slow_wave_below_6"] += 1
                else:
                    counts["n3_slow_wave_at_least_6"] += 1
    n3_support = counts["reference_N3"]
    return {
        "counts": dict(counts),
        "maxima_seconds": maxima,
        "n3_coverage": {
            "support": n3_support,
            "no_slow_wave_fraction": counts["n3_no_slow_wave"] / n3_support if n3_support else None,
            "below_6_seconds_fraction": counts["n3_slow_wave_below_6"] / n3_support if n3_support else None,
            "at_least_6_seconds_fraction": counts["n3_slow_wave_at_least_6"] / n3_support if n3_support else None,
        },
        "interpretation_boundary": (
            "These are frozen evidence reachability counts, not event-level false-negative rates."
        ),
    }
