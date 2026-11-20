from __future__ import annotations

import json
from pathlib import Path

from aasm_v4.contracts import validate_handler_registry
from aasm_v4.profiles import load_detector_profile, load_rule_profile, repository_root
from aasm_v4.rules import C196_POLICY


def test_detector_champion_profile() -> None:
    baseline = load_detector_profile("B000")
    champion = load_detector_profile("M6408")
    assert baseline.enabled_handlers == ()
    assert champion.parent_profile == "B000"
    assert champion.enabled_handlers == ("E04", "E11", "SP", "AR")


def test_c196_changes_only_two_modules() -> None:
    profile = load_rule_profile("C196")
    assert profile.parent_config == "temporal_r1"
    assert profile.changed_modules == ("R.BACKWARD_BRIDGE", "N2_POST_AROUSAL_REENTRY")
    assert profile.policies["rem_backward_policy"] == "aasm_v21_backward_bridge"
    assert C196_POLICY["post_arousal_reentry_policy"] == "structure_reentry_first"


def test_handler_registry_is_complete_and_bounded() -> None:
    path = repository_root() / "configs" / "v4" / "handlers.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    validate_handler_registry(payload["handlers"])
    by_id = {row["handler_id"]: row for row in payload["handlers"]}
    assert by_id["E13"]["evidence_level"] == "movement_proxy"
    assert by_id["E07"]["evidence_level"] == "weak_supervision_only"
    assert by_id["E02"]["known_downstream_result"]


def test_public_configs_do_not_name_records() -> None:
    root = repository_root() / "configs" / "v4"
    for path in root.rglob("*.json"):
        text = path.read_text(encoding="utf-8")
        assert "SN001" not in text
        assert "SN010" not in text
        assert "C:\\Users" not in text
        assert "D:\\EEG_Conformer" not in text
