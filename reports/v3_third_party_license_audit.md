# V3 third-party license and distribution audit

This branch is a public research release under the repository's existing
[GPLv3 license](../LICENSE). The table records the historical research
dependencies and the narrower set actually distributed here.

| Component | Verified identity | Observed license | Distribution decision |
|---|---|---|---|
| PSGScoring source | `bartromb/psgscoring` commit `7271cb089feacb585332ed0db8a375019548702d` | BSD-3-Clause in the sealed source audit | Link and adapter contract only. Source and ONNX weights are not copied. Weight redistribution permission was not treated as established. |
| PSGScoring U-Net v1 ONNX | SHA256 `6fbea2857567f95023e6dd34461fba92cab238c9c334dbe9d66494b258aa715f` | Model artifact provenance recorded; redistribution scope not assumed | Not distributed. Users must obtain any permitted artifact separately and verify the hash. |
| ScoringHero MT-KCD | commit `ff47238fd9c18e779ba1420d8b8b1cdf8940c19e`; source SHA256 `0cd1173672685bd88c0c8a3eb8498efebcd27a8c2c883813bdbed3116e548ea8` | AGPL-3.0-only; LICENSE SHA256 `e0eedba615d5cd1b986afb6c5b3a4b1ae33713e7e9dc74d19daec5e3221f9d2e` | No third-party source copied. Only a detector-neutral canonical-event contract is published. |
| Wonambi | installed version 7.17 | Installed distribution metadata reported GPLv3, while a bundled audit reported BSD-3-Clause | License metadata conflict remains unresolved; package/source is not vendored. |
| YASA | frozen V2 dependency | See [`references/YASA_LICENSE.txt`](../references/YASA_LICENSE.txt) | Existing frozen dependency only; no repackaged source or weights. |
| SleepFM | `zou-group/sleepfm-clinical` commit `2bcbae04c3592f61352addb7ac3d4193f0a3ca25` | CC BY-NC 4.0 in the sealed source audit | Negative aggregate result only; no source/checkpoint distributed. |

## Public code scope

[`experiments/v3_detector_integration`](../experiments/v3_detector_integration)
contains only self-authored, detector-neutral code:

- canonical event/provenance validation;
- deterministic `REPLACE / KEEP_FROZEN / UNAVAILABLE` policy;
- copy-on-write overlay;
- K-complex/arousal association rebuild using the frozen timing contract;
- synthetic fixtures and contract tests.

It contains no implementation copied from PSGScoring, ScoringHero, Wonambi,
YASA, or SleepFM. It also contains no model weight, clinical signal, patient
identifier, stage label, or per-epoch prediction.

## Residual uncertainty

This is an engineering distribution audit, not legal advice. In particular,
the Wonambi metadata conflict and PSGScoring weight permission should be
resolved with the upstream maintainers before any future redistribution. The
safe default used here is non-distribution.
