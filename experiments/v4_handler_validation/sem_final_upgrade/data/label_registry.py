from __future__ import annotations

from pathlib import Path
from typing import Any

from ..common import sha256


def build_label_provenance(config: dict[str, Any]) -> dict[str, Any]:
    root = Path(config["frozen_heritage_root"])
    source = root / "src" / "aasm_v2" / "events" / "morphology.py"
    preprocessing = root / "src" / "aasm_v2" / "preprocessing" / "__init__.py"
    settings = root / "configs" / "frozen_detector_config.json"
    return {
        "schema_version": 1,
        "label_source": "WEAK_TEACHER_SEM",
        "expert_sem_event_count": 0,
        "expert_sem_dataset_search": "NO_PUBLIC_DOWNLOADABLE_EVENT_DATA_VERIFIED",
        "stage_labels_used": False,
        "teacher_semantics": (
            "Frozen V3 dual-EOG morphology: 0.3-5 Hz anti-phase deflection, "
            "30 uV minimum, correlation <= -0.5, 1-5 s duration, "
            "initial deflection >= 0.5 s"
        ),
        "teacher_source": str(source.resolve()),
        "teacher_source_sha256": sha256(source),
        "preprocessing_source": str(preprocessing.resolve()),
        "preprocessing_source_sha256": sha256(preprocessing),
        "teacher_config": str(settings.resolve()),
        "teacher_config_sha256": sha256(settings),
        "evaluation_interpretation": (
            "Teacher fidelity on record-disjoint real EOG; not expert SEM accuracy"
        ),
        "evaluation_domain": (
            "All bilateral-EOG QC-valid samples; no target-derived mask is supplied "
            "to eventization"
        ),
    }
