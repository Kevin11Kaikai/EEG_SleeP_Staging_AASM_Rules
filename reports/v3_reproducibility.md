# V3 reproducibility and release scope

## Reproducibility layers

| Layer | Publicly reproducible from this branch? | Notes |
|---|---|---|
| Frozen V2 / Temporal R1 source and synthetic tests | Yes | Inherited byte-for-byte from base commit `155baa7895c2a4a20f8a788901f8412c3f8780b3`. |
| Seven educational figures | Yes | Generated from one public aggregate JSON with fixed seeds; SVG and PNG are committed. |
| Overlay and K–arousal association contracts | Yes | Detector-neutral code and synthetic fixtures only. |
| Published aggregate metrics | Auditable | Logical artifact identities and source SHA256 values are public in `v3_evidence_summary.json`. |
| Historical event matching and ten-night eight-arm replay | Local sealed evidence required | Patient/event timelines and predictions are intentionally not public. |
| Fresh raw-EDF end-to-end detector run | No | The required SN001 smoke did not finish after 8,352 seconds. |
| Optional third-party models | No | Weights/source are not redistributed; users must follow upstream licenses. |

## Public setup and tests

The frozen V2 project targets the dependency set in
[`requirements.lock.txt`](../requirements.lock.txt). A clean environment can be
prepared with:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.lock.txt
.\.venv\Scripts\python.exe -m pip install --no-deps -e .
$env:PYTHONDONTWRITEBYTECODE='1'
$env:PYTHONUTF8='1'
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider
```

Run the new public contract tests without any clinical data:

```powershell
python -m pytest tests/test_v3_detector_integration.py -q
```

Regenerate all seven figures:

```powershell
$env:MPLBACKEND='Agg'
python scripts/generate_v3_educational_images.py
```

Run the release validator:

```powershell
python scripts/validate_v3_release.py --write-report
```

The executed test matrix, including the dependency-only failure in the system
Python environment and the passing frozen research environment, is recorded in
[`v3_test_results.json`](v3_test_results.json). The incomplete environment is
preserved as evidence rather than being reclassified as a passing run.

## Frozen heritage identity

- Base branch: `AASM_Rules_V2`
- Base commit: `155baa7895c2a4a20f8a788901f8412c3f8780b3`
- Frozen V2 source hash:
  `e47d39fc6f58c8d009af9531f197a66f3d0ce8e28b5b2e37983de1747c121261`
- Temporal R1 hash:
  `c92de5185a049f051df4f8aebd93baa4fcd62b051f9f6de4271642de99951d78`
- Legacy main rule-flow commit actually read for this release:
  `9dd964c2ecdcd1ab2b305278876598d3bccbc014`
- Older main commit recorded by the inherited V2 reference file:
  `8e973edb85eb4b9a9cadcb3adbb855b484d477c1`

The last two are intentionally separate: the inherited reference is historical,
while the later commit is the remote main state inspected during V3 release
preflight.

## Label and privacy boundary

The release contains no EDF/BDF, scoring file, individual event timeline,
patient-level metric, per-epoch prediction, clinical waveform, detailed trace,
cached model input, model weight, or local absolute path. Synthetic signals in
the figures are generated in code and explicitly labeled.

The prior HMC computation opened labels only in its evaluation process after
predictions had been sealed. This documentation release does not rerun the HMC
pipeline and does not access final141.

## How to connect a local detector legally

1. Obtain the detector and any weights from the upstream source under its
   license.
2. Verify the pinned version/checkpoint hash.
3. Produce canonical events satisfying
   [`contracts.py`](../experiments/v3_detector_integration/contracts.py).
4. Record `REPLACE`, `KEEP_FROZEN`, or `UNAVAILABLE` for A/K/S explicitly.
5. Apply the copy-on-write overlay and rebuild K–arousal associations.
6. Keep all patient-level banks, evidence, predictions, and traces outside the
   public repository.

This interface makes source identity and replacement semantics auditable; it
does not turn an unavailable third-party model into a bundled dependency.
