from experiments.v4_mechanistic_attribution.core import metric_from_arrays


def test_abstention_counts_as_fixed_mask_error_but_reduces_coverage() -> None:
    metric = metric_from_arrays(["W", "N2"], ["W", "UNDETERMINED"])
    assert metric["accuracy"] == 0.5
    assert metric["coverage"] == 0.5
    assert metric["conditional_accuracy"] == 1.0
    assert metric["per_stage"]["N2"]["fn"] == 1
