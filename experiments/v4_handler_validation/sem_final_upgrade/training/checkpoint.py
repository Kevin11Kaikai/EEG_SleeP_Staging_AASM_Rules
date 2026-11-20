from __future__ import annotations

from pathlib import Path

import torch

from ..models import build_model


def strict_load(checkpoint: str | Path, architecture: str, device: torch.device):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = build_model(architecture)
    incompatible = model.load_state_dict(payload["state_dict"], strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError("Strict checkpoint load reported incompatible keys")
    model.to(device).eval()
    return model, payload
