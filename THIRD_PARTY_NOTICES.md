# Third-party notices and non-redistribution policy

This repository is distributed under its existing GPLv3 license. That license
does not grant rights to third-party datasets, model weights, papers, or
software. The V4 branch publishes adapters, provenance, and aggregate research
results; it does not redistribute the external artifacts listed below.

| Dependency or source | Public locator | Observed status | V4 distribution |
|---|---|---|---|
| PSGScoring U-Net | [bartromb/psgscoring](https://github.com/bartromb/psgscoring) | Source audit recorded BSD-3-Clause; weight redistribution was not assumed | Adapter/provenance only; no source copy or weights |
| SSSM | [PyPI package](https://pypi.org/project/sssm/) | Upstream package metadata applies | Interface/provenance only; no model artifact |
| MT-KCD / ScoringHero | Upstream project recorded in V3 audit | AGPL-3.0-only source | No vendored code or weights |
| Wonambi | Upstream project recorded in V3 audit | Conflicting license metadata was observed in the sealed audit | No vendored code or weights |
| SUMOv2 Spindle | Source identity recorded in the sealed study manifest | Checkpoint redistribution permission not established | Aggregate results and thin contract only |
| K08 and R04 | Locally trained research models | Project research artifacts; clinical redistribution not authorized | Configuration identity and aggregate results only |
| Phase 1C/1D temporal handlers | Locally trained, mostly weak-teacher research models | No independent expert-event validation for most handlers | Architecture/protocol only; no checkpoints |

## Data

- DREAMS source material was treated as CC BY-NC-ND 3.0 in the project data
  audit. No signals, labels, windows, event lists, or waveform overlays are
  redistributed here.
- HMC and PSG-IPA raw data, labels, metadata, and patient-level derived
  timelines are not distributed.
- Users must obtain every dataset from its official source and independently
  confirm the current license and intended use.

## AASM material

This branch implements project-specific, executable operationalizations of
public sleep-scoring concepts. It does not reproduce substantial text from the
AASM Manual and must not be described as AASM-certified or AASM-compliant
clinical software.

## Model-artifact policy

Files ending in `.pt`, `.pth`, `.ckpt`, `.onnx`, or
`.safetensors` are intentionally excluded. A SHA256 identity in a manifest is
provenance, not permission to redistribute the corresponding artifact.
