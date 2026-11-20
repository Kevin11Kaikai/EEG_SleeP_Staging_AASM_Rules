from __future__ import annotations

import csv
import json
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader

from .common import (
    binary_counts,
    parameter_hash,
    set_seed,
    sha256,
    state_l2_delta,
    write_json,
)
from .data import TemporalWindowDataset, head_prevalence
from .inference import infer_full_record
from .models import FAMILY_SPECS, build_model


def _masked_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    positive_weight: torch.Tensor,
) -> torch.Tensor:
    raw = nn.functional.binary_cross_entropy_with_logits(
        logits,
        target,
        reduction="none",
        pos_weight=positive_weight.view(1, -1, 1),
    )
    denominator = torch.clamp(valid.sum(), min=1.0)
    return torch.sum(raw * valid) / denominator


def _collect_validation(
    model: nn.Module, loader: DataLoader, device: torch.device, positive_weight: torch.Tensor
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    probabilities = []
    targets = []
    validity = []
    losses = []
    with torch.inference_mode():
        for signal, target, valid, _, _ in loader:
            signal = signal.to(device)
            target_device = target.to(device)
            valid_device = valid.to(device)
            logits = model(signal)
            losses.append(
                float(_masked_loss(logits, target_device, valid_device, positive_weight))
            )
            probabilities.append(torch.sigmoid(logits).cpu().numpy())
            targets.append(target.numpy())
            validity.append(valid.numpy())
    return (
        float(np.mean(losses)),
        np.concatenate(probabilities, axis=0),
        np.concatenate(targets, axis=0),
        np.concatenate(validity, axis=0),
    )


def _select_thresholds(
    probabilities: np.ndarray,
    targets: np.ndarray,
    validity: np.ndarray,
    heads: tuple[str, ...],
    grid: list[float],
) -> tuple[dict[str, float], dict[str, dict[str, float | int]]]:
    thresholds = {}
    metrics = {}
    for index, head in enumerate(heads):
        best_threshold = 0.5
        best_metrics = None
        for threshold in grid:
            row = binary_counts(
                targets[:, index].reshape(-1),
                probabilities[:, index].reshape(-1) >= threshold,
                validity[:, index].reshape(-1),
            )
            if best_metrics is None or (row["f1"], row["precision"], -threshold) > (
                best_metrics["f1"],
                best_metrics["precision"],
                -best_threshold,
            ):
                best_metrics = row
                best_threshold = threshold
        thresholds[head] = float(best_threshold)
        metrics[head] = best_metrics or {}
    return thresholds, metrics


def _write_history(path: Path, history: list[dict[str, Any]]) -> None:
    keys = sorted({key for row in history for key in row})
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(history)


def _train_negative_control(
    family: str,
    signals: dict[str, np.ndarray],
    labels: dict[str, dict[str, np.ndarray]],
    train_ids: list[str],
    validation_loader: DataLoader,
    family_config: dict[str, Any],
    positive_weight: torch.Tensor,
    epochs: int,
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    set_seed(seed + 1000)
    model = build_model(family).to(device)
    shuffled = TemporalWindowDataset(
        signals,
        labels,
        family,
        train_ids,
        int(family_config["window_seconds"]),
        int(family_config["stride_seconds"]),
        permute_targets=True,
        seed=seed + 1000,
    )
    loader = DataLoader(
        shuffled,
        batch_size=int(family_config["batch_size"]),
        shuffle=True,
        num_workers=0,
        generator=torch.Generator().manual_seed(seed + 1000),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=float(family_config["learning_rate"]))
    steps = 0
    for _ in range(epochs):
        model.train()
        for signal, target, valid, _, _ in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = _masked_loss(
                model(signal.to(device)), target.to(device), valid.to(device), positive_weight
            )
            if not torch.isfinite(loss):
                raise FloatingPointError("Label-shuffle control produced non-finite loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            steps += 1
    loss, probability, target, valid = _collect_validation(
        model, validation_loader, device, positive_weight
    )
    metrics = {
        head: binary_counts(
            target[:, index].reshape(-1),
            probability[:, index].reshape(-1) >= 0.5,
            valid[:, index].reshape(-1),
        )
        for index, head in enumerate(FAMILY_SPECS[family].output_heads)
    }
    return {
        "kind": "LABEL_SHUFFLE_NEGATIVE_CONTROL",
        "epochs": epochs,
        "optimizer_steps": steps,
        "validation_loss": loss,
        "fixed_threshold": 0.5,
        "metrics": metrics,
    }


def train_family(
    run_dir: Path,
    family: str,
    signals: dict[str, np.ndarray],
    labels: dict[str, dict[str, np.ndarray]],
    split: dict[str, list[str]],
    config: dict[str, Any],
) -> dict[str, Any]:
    spec = FAMILY_SPECS[family]
    family_config = config["families"][family]
    model_dir = run_dir / "models" / family
    model_dir.mkdir(parents=True, exist_ok=True)
    write_json(model_dir / "model_config.json", {**spec.to_dict(), **family_config})
    seed = int(config["seed"])
    set_seed(seed)
    train_dataset = TemporalWindowDataset(
        signals,
        labels,
        family,
        split["train"],
        int(family_config["window_seconds"]),
        int(family_config["stride_seconds"]),
        seed=seed,
    )
    validation_dataset = TemporalWindowDataset(
        signals,
        labels,
        family,
        split["validation"],
        int(family_config["window_seconds"]),
        int(family_config["window_seconds"]),
        seed=seed,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(family_config["batch_size"]),
        shuffle=True,
        num_workers=0,
        generator=torch.Generator().manual_seed(seed),
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=int(family_config["batch_size"]),
        shuffle=False,
        num_workers=0,
    )
    prevalence = head_prevalence(labels, split["train"], spec.output_heads)
    weights = []
    for head in spec.output_heads:
        positive = int(prevalence[head]["positive_samples"])
        valid = int(prevalence[head]["valid_samples"])
        if positive == 0:
            raise ValueError(f"No train-positive weak labels for {head}")
        weights.append(min(20.0, max(1.0, (valid - positive) / positive)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    positive_weight = torch.tensor(weights, device=device)
    model = build_model(family).to(device)
    initial_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    initial_hash = parameter_hash(model)
    trainable_parameters = sum(value.numel() for value in model.parameters() if value.requires_grad)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(family_config["learning_rate"]), weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=1
    )
    history = []
    best_score = -1.0
    best_epoch = -1
    best_thresholds = {}
    best_metrics = {}
    epochs_without_gain = 0
    optimizer_steps = 0
    nonfinite_gradients = 0
    started = time.perf_counter()
    gpu_seconds = 0.0
    checkpoint = model_dir / "best_checkpoint.pt"
    for epoch in range(1, int(family_config["max_epochs"]) + 1):
        epoch_start = time.perf_counter()
        model.train()
        train_losses = []
        for signal, target, valid, _, _ in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(signal.to(device))
            loss = _masked_loss(logits, target.to(device), valid.to(device), positive_weight)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite training loss for {family}")
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            if not torch.isfinite(grad_norm):
                nonfinite_gradients += 1
                optimizer.zero_grad(set_to_none=True)
                continue
            optimizer.step()
            optimizer_steps += 1
            train_losses.append(float(loss.detach().cpu()))
        if device.type == "cuda":
            torch.cuda.synchronize()
            gpu_seconds += time.perf_counter() - epoch_start
        validation_loss, probability, target, valid = _collect_validation(
            model, validation_loader, device, positive_weight
        )
        thresholds, metrics = _select_thresholds(
            probability,
            target,
            valid,
            spec.output_heads,
            [float(value) for value in config["threshold_grid"]],
        )
        score = float(np.mean([metrics[head]["f1"] for head in spec.output_heads]))
        scheduler.step(score)
        row: dict[str, Any] = {
            "epoch": epoch,
            "train_loss": float(np.mean(train_losses)),
            "validation_loss": validation_loss,
            "mean_validation_f1": score,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "optimizer_steps_cumulative": optimizer_steps,
        }
        for head in spec.output_heads:
            row[f"{head}_f1"] = metrics[head]["f1"]
            row[f"{head}_precision"] = metrics[head]["precision"]
            row[f"{head}_recall"] = metrics[head]["recall"]
            row[f"{head}_threshold"] = thresholds[head]
        history.append(row)
        if score > best_score + 1e-8:
            best_score = score
            best_epoch = epoch
            best_thresholds = thresholds
            best_metrics = metrics
            epochs_without_gain = 0
            torch.save(
                {
                    "schema_version": 1,
                    "family": family,
                    "model_spec": spec.to_dict(),
                    "state_dict": model.state_dict(),
                    "selected_thresholds": thresholds,
                    "epoch": epoch,
                    "seed": seed,
                },
                checkpoint,
            )
        else:
            epochs_without_gain += 1
        if (
            epoch >= int(family_config["minimum_epochs"])
            and epochs_without_gain >= int(family_config["patience"])
        ):
            break
    elapsed = time.perf_counter() - started
    if optimizer_steps < 10 or not checkpoint.exists():
        raise RuntimeError(f"Insufficient completed training for {family}")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    strict_model = build_model(family)
    incompatible = strict_model.load_state_dict(payload["state_dict"], strict=True)
    strict_model.eval()
    final_hash = parameter_hash(strict_model)
    delta = state_l2_delta(initial_state, strict_model.state_dict())
    _write_history(model_dir / "training_history.csv", history)
    negative_control = _train_negative_control(
        family,
        signals,
        labels,
        split["train"],
        validation_loader,
        family_config,
        positive_weight,
        int(config["negative_control_epochs"]),
        seed,
        device,
    )
    checkpoint_manifest = {
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256(checkpoint),
        "initial_parameter_hash": initial_hash,
        "final_parameter_hash": final_hash,
        "weight_delta_l2": delta,
        "strict_load_missing_keys": list(incompatible.missing_keys),
        "strict_load_unexpected_keys": list(incompatible.unexpected_keys),
        "best_epoch": best_epoch,
        "selected_thresholds": best_thresholds,
        "trainable_parameter_count": trainable_parameters,
        "actual_optimizer_steps": optimizer_steps,
        "actual_epochs": len(history),
        "training_wall_seconds": elapsed,
        "actual_gpu_seconds": gpu_seconds,
        "nonfinite_gradient_steps_skipped": nonfinite_gradients,
        "device": str(device),
        "training_label_source": (
            "PHYSIOLOGICAL_CONSTRAINT_WEAK_LABEL"
            if family == "major_movement"
            else "FROZEN_V3_WEAK_TEACHER"
        ),
        "test_labels_used_for_training_or_selection": False,
    }
    write_json(model_dir / "checkpoint_manifest.json", checkpoint_manifest)
    write_json(
        model_dir / "validation_metrics.json",
        {
            "best_validation_mean_f1": best_score,
            "per_head": best_metrics,
            "train_prevalence": prevalence,
            "label_shuffle_negative_control": negative_control,
        },
    )
    return {
        "family": family,
        "checkpoint": checkpoint,
        "checkpoint_manifest": checkpoint_manifest,
        "selected_thresholds": best_thresholds,
        "validation_metrics": best_metrics,
        "negative_control": negative_control,
        "device": str(device),
    }
