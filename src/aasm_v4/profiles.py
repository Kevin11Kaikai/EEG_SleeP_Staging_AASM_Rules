"""Load portable V4 detector and rule profiles from ``configs/v4``."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .contracts import DetectorProfile, RuleProfile


def repository_root() -> Path:
    """Return the source-checkout root without relying on a machine-specific path."""

    for parent in Path(__file__).resolve().parents:
        if (parent / "configs" / "v4").is_dir():
            return parent
    raise RuntimeError("could not locate configs/v4 from the installed source tree")


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object: {path}")
    return value


def load_detector_profile(profile_id: str) -> DetectorProfile:
    """Load ``b000`` or ``m6408`` from the public detector-profile directory."""

    safe_id = profile_id.lower()
    if not safe_id.replace("_", "").isalnum():
        raise ValueError("profile_id contains unsupported characters")
    path = repository_root() / "configs" / "v4" / "detector_profiles" / f"{safe_id}.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    return DetectorProfile.from_mapping(_read_json(path))


def load_rule_profile(config_id: str) -> RuleProfile:
    """Load a public rule profile such as ``temporal_r1`` or ``c196``."""

    safe_id = config_id.lower()
    if not safe_id.replace("_", "").isalnum():
        raise ValueError("config_id contains unsupported characters")
    path = repository_root() / "configs" / "v4" / "rule_profiles" / f"{safe_id}.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    return RuleProfile.from_mapping(_read_json(path))
