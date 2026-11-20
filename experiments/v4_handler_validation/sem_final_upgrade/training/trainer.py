from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..common import (
    parameter_hash,
    set_seed,
    sha256,
    state_l2_delta,
    write_csv,
    write_json,
)
from ..data.audit import RecordData
from ..evaluation.challenge import infer_full_record, select_validation_threshold
from ..models import build_model
from .losses import masked_bce_dice_loss
from .sampler import SEMWindowDataset, build_weighted_sampler


def _record_probabilities(
    model: torch.nn.Module,
    records: dict[str, RecordData],
    record_ids: list[str],
    device: torch.device,
    model_config: dict[str, Any],
) -> dict[str, np.ndarray]:
    return {
        record_id: infer_full_record(
            model,
            records[record_id].signal,
            device,
            int(model_config["window_seconds"]),
            int(model_config["inference_stride_seconds"]),
            int(model_config["batch_size"]),
        )
        for record_id in record_ids
    }


def _full_record_validation_loss(
    records: dict[str, RecordData],
    record_ids: list[str],
    probabilities: dict[str, np.ndarray],
    positive_weight: float,
    dice_weight: float,
) -> float:
    positive_probability: list[np.ndarray] = []
    negative_probability: list[np.ndarray] = []
    dice_numerator = 1.0
    dice_denominator = 1.0
    for record_id in record_ids:
        record = records[record_id]
        valid = record.scoring_mask.astype(bool)
        target = record.target.astype(bool)
        probability = np.clip(probabilities[record_id], 1e-7, 1.0 - 1e-7)
        positive_probability.append(probability[valid & target])
        negative_probability.append(probability[valid & ~target])
        dice_numerator += 2.0 * float(np.sum(probability[valid] * target[valid]))
        dice_denominator += float(np.sum(probability[valid]) + np.sum(target[valid]))
    positive = np.concatenate([value for value in positive_probability if value.size])
    negative = np.concatenate([value for value in negative_probability if value.size])
    positive_loss = float(-np.log(positive).mean()) if positive.size else 0.0
    negative_loss = float(-np.log1p(-negative).mean()) if negative.size else 0.0
    dice_loss = 1.0 - dice_numerator / dice_denominator
    return positive_weight * positive_loss + negative_loss + dice_weight * dice_loss


def _train_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    model_config: dict[str, Any],
) -> tuple[float, int, int]:
    model.train()
    losses = []
    steps = 0
    nonfinite = 0
    for signal, target, valid, _, _ in loader:
        optimizer.zero_grad(set_to_none=True)
        logits = model(signal.to(device))
        loss = masked_bce_dice_loss(
            logits,
            target.to(device),
            valid.to(device),
            float(model_config["positive_loss_weight"]),
            float(model_config["dice_weight"]),
        )
        if not torch.isfinite(loss):
            nonfinite += 1
            continue
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        if not torch.isfinite(grad_norm):
            optimizer.zero_grad(set_to_none=True)
            nonfinite += 1
            continue
        optimizer.step()
        steps += 1
        losses.append(float(loss.detach().cpu()))
    if not losses:
        raise RuntimeError("No finite SEM optimizer steps in epoch")
    return float(np.mean(losses)), steps, nonfinite


def _label_shuffle_control(
    records: dict[str, RecordData],
    train_ids: list[str],
    validation_ids: list[str],
    config: dict[str, Any],
    device: torch.device,
    epochs: int,
) -> dict[str, Any]:
    model_config = config["model"]
    seed = int(config["seed"]) + 9001
    set_seed(seed)
    dataset = SEMWindowDataset(
        records,
        train_ids,
        int(model_config["window_seconds"]),
        int(model_config["train_stride_seconds"]),
        shuffled_targets=True,
        seed=seed,
    )
    sampler = build_weighted_sampler(
        dataset,
        float(model_config["positive_window_weight"]),
        float(model_config["hard_negative_window_weight"]),
        seed,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(model_config["batch_size"]),
        sampler=sampler,
        num_workers=0,
    )
    model = build_model(str(model_config["architecture"])).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(model_config["learning_rate"]),
        weight_decay=float(model_config["weight_decay"]),
    )
    steps = 0
    losses = []
    for _ in range(epochs):
        loss, count, _ = _train_epoch(model, loader, optimizer, device, model_config)
        losses.append(loss)
        steps += count
    probabilities = _record_probabilities(
        model, records, validation_ids, device, model_config
    )
    threshold, result, grid = select_validation_threshold(
        records,
        validation_ids,
        probabilities,
        [float(value) for value in config["selection"]["threshold_grid"]],
        config["eventization"],
        float(config["selection"]["minimum_iou"]),
    )
    return {
        "kind": "LABEL_SHUFFLE_NEGATIVE_CONTROL",
        "seed": seed,
        "epochs": epochs,
        "optimizer_steps": steps,
        "final_train_loss": losses[-1],
        "selected_validation_threshold": threshold,
        "validation_metrics": result["aggregate"],
        "threshold_grid": grid,
        "checkpoint_saved": False,
    }


def train_sem_model(
    run_dir: Path,
    records: dict[str, RecordData],
    split_manifest: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    model_config = config["model"]
    architecture = str(model_config["architecture"])
    train_ids = list(split_manifest["records"]["train"])
    validation_ids = list(split_manifest["records"]["validation"])
    model_dir = run_dir / "models" / ("tcn" if architecture == "inherited_tcn" else architecture)
    model_dir.mkdir(parents=True, exist_ok=True)
    seed = int(config["seed"])
    set_seed(seed)
    dataset = SEMWindowDataset(
        records,
        train_ids,
        int(model_config["window_seconds"]),
        int(model_config["train_stride_seconds"]),
        seed=seed,
    )
    sampler = build_weighted_sampler(
        dataset,
        float(model_config["positive_window_weight"]),
        float(model_config["hard_negative_window_weight"]),
        seed,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(model_config["batch_size"]),
        sampler=sampler,
        num_workers=0,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(architecture).to(device)
    initial_state = {
        key: value.detach().cpu().clone() for key, value in model.state_dict().items()
    }
    initial_hash = parameter_hash(model)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(model_config["learning_rate"]),
        weight_decay=float(model_config["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2
    )
    checkpoint = model_dir / "best_checkpoint.pt"
    history = []
    best_key = None
    best_epoch = 0
    best_threshold = None
    best_validation = None
    epochs_without_gain = 0
    optimizer_steps = 0
    nonfinite_steps = 0
    started = time.perf_counter()
    gpu_seconds = 0.0
    for epoch in range(1, int(model_config["maximum_epochs"]) + 1):
        epoch_start = time.perf_counter()
        train_loss, steps, nonfinite = _train_epoch(
            model, loader, optimizer, device, model_config
        )
        if device.type == "cuda":
            torch.cuda.synchronize()
            gpu_seconds += time.perf_counter() - epoch_start
        optimizer_steps += steps
        nonfinite_steps += nonfinite
        probabilities = _record_probabilities(
            model, records, validation_ids, device, model_config
        )
        validation_loss = _full_record_validation_loss(
            records,
            validation_ids,
            probabilities,
            float(model_config["positive_loss_weight"]),
            float(model_config["dice_weight"]),
        )
        threshold, validation, threshold_grid = select_validation_threshold(
            records,
            validation_ids,
            probabilities,
            [float(value) for value in config["selection"]["threshold_grid"]],
            config["eventization"],
            float(config["selection"]["minimum_iou"]),
        )
        aggregate = validation["aggregate"]
        score = float(aggregate["f1"] or 0.0)
        key = (
            score,
            float(aggregate["precision"]),
            -float(aggregate["fp_per_hour"] or 0.0),
        )
        scheduler.step(score)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
                "validation_event_f1": aggregate["f1"],
                "validation_event_precision": aggregate["precision"],
                "validation_event_recall": aggregate["recall"],
                "validation_fp_per_hour": aggregate["fp_per_hour"],
                "validation_tp": aggregate["tp"],
                "validation_fp": aggregate["fp"],
                "validation_fn": aggregate["fn"],
                "selected_threshold": threshold,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "optimizer_steps_cumulative": optimizer_steps,
            }
        )
        if best_key is None or key > best_key:
            best_key = key
            best_epoch = epoch
            best_threshold = threshold
            best_validation = validation
            epochs_without_gain = 0
            torch.save(
                {
                    "schema_version": 1,
                    "architecture": architecture,
                    "state_dict": model.state_dict(),
                    "epoch": epoch,
                    "seed": seed,
                    "selected_threshold": threshold,
                    "eventization": config["eventization"],
                    "selection_metric": "validation full-record event F1, IoU>=0.2",
                },
                checkpoint,
            )
            write_json(model_dir / "best_threshold_grid.json", threshold_grid)
        else:
            epochs_without_gain += 1
        if (
            epoch >= int(model_config["minimum_epochs"])
            and epochs_without_gain >= int(model_config["patience"])
        ):
            break
    wall_seconds = time.perf_counter() - started
    if not checkpoint.exists() or optimizer_steps < 10:
        raise RuntimeError("SEM training did not produce a valid checkpoint")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    strict_model = build_model(architecture)
    incompatible = strict_model.load_state_dict(payload["state_dict"], strict=True)
    strict_model.eval()
    final_hash = parameter_hash(strict_model)
    delta = state_l2_delta(initial_state, strict_model.state_dict())
    write_csv(model_dir / "training_history.csv", history)
    manifest = {
        "architecture": architecture,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256(checkpoint),
        "initial_parameter_hash": initial_hash,
        "final_parameter_hash": final_hash,
        "weight_delta_l2": delta,
        "strict_load_missing_keys": list(incompatible.missing_keys),
        "strict_load_unexpected_keys": list(incompatible.unexpected_keys),
        "model_parameters": sum(value.numel() for value in strict_model.parameters()),
        "trainable_parameters": sum(
            value.numel() for value in strict_model.parameters() if value.requires_grad
        ),
        "actual_optimizer_steps": optimizer_steps,
        "actual_epochs": len(history),
        "best_epoch": best_epoch,
        "selected_threshold": best_threshold,
        "training_wall_seconds": wall_seconds,
        "gpu_training_seconds": gpu_seconds,
        "nonfinite_steps_skipped": nonfinite_steps,
        "device": str(device),
        "training_label_source": "WEAK_TEACHER_SEM",
        "expert_or_stage_labels_used": False,
        "positive_window_oversampling_train_only": True,
        "test_or_challenge_used_for_selection": False,
    }
    write_json(model_dir / "checkpoint_manifest.json", manifest)
    write_json(
        model_dir / "validation_metrics.json",
        {
            "best_epoch": best_epoch,
            "selected_threshold": best_threshold,
            "metrics": best_validation["aggregate"] if best_validation else None,
            "per_record": best_validation["per_record"] if best_validation else [],
        },
    )
    control = _label_shuffle_control(
        records,
        train_ids,
        validation_ids,
        config,
        device,
        max(int(model_config["label_shuffle_epochs"]), len(history)),
    )
    return {
        "architecture": architecture,
        "checkpoint": checkpoint,
        "checkpoint_manifest": manifest,
        "selected_threshold": float(best_threshold),
        "best_validation": best_validation,
        "label_shuffle_control": control,
        "device": device,
    }
