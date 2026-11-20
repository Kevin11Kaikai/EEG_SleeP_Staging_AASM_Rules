from __future__ import annotations

import torch
from torch import nn


def masked_bce_dice_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    positive_weight: float,
    dice_weight: float,
) -> torch.Tensor:
    positive = (target > 0.5) & (valid > 0.5)
    negative = (target <= 0.5) & (valid > 0.5)
    positive_loss = -nn.functional.logsigmoid(logits[positive]).mean()
    negative_loss = -nn.functional.logsigmoid(-logits[negative]).mean()
    if not torch.isfinite(positive_loss):
        positive_loss = torch.zeros((), device=logits.device)
    if not torch.isfinite(negative_loss):
        negative_loss = torch.zeros((), device=logits.device)
    bce = positive_weight * positive_loss + negative_loss
    probability = torch.sigmoid(logits) * valid
    target_valid = target * valid
    intersection = torch.sum(probability * target_valid)
    dice = 1.0 - (2.0 * intersection + 1.0) / (
        torch.sum(probability) + torch.sum(target_valid) + 1.0
    )
    return bce + dice_weight * dice
