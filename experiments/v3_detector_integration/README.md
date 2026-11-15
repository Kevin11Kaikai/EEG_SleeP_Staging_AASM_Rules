# Public V3 detector-integration contracts

This directory contains the publishable, self-authored part of the historical
event-detector integration: canonical proposal validation, copy-on-write event
replacement, deterministic K-complex/arousal association rebuilding, and
synthetic contract tests.

It deliberately does **not** contain PSG, labels, model weights, third-party
detector source, clinical traces, or a fresh raw-EDF runner.  A detector-specific
adapter must produce canonical events with stable IDs, provenance, QC status,
and signal sample pointers.  The overlay then replaces exactly the requested
event type; it never silently unions old and new proposals.

The historical HMC computation used sealed event banks and ran all eight A/K/S
factorial arms.  The fresh SN001 base detector smoke did not finish after 8,352
seconds, so this public layer is an auditable integration contract—not a claim
that fresh raw-EDF end-to-end reproduction has passed.

Run the synthetic tests from the repository root:

```powershell
python -m pytest tests/test_v3_detector_integration.py -q
```

See [the license audit](../../reports/v3_third_party_license_audit.md) before
connecting any optional third-party detector.
