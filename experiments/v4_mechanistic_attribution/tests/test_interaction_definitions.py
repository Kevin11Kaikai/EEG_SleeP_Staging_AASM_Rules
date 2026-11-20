def test_group_nonadditivity_definition() -> None:
    baseline = 0.50
    group = 0.53
    singles = [0.51, 0.49, 0.52]
    observed = group - baseline
    summed = sum(value - baseline for value in singles)
    assert abs((observed - summed) - 0.01) < 1e-12
