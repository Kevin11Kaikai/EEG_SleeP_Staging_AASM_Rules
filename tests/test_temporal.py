"""Source-grounded timing contracts; labels here are synthetic expected behavior."""
from copy import deepcopy
import inspect
import json
from pathlib import Path
import hashlib
import pytest
from aasm_v2.evidence import aggregate, ROLES
from aasm_v2.rules import enhanced_engine, inherit
from aasm_temporal_r1 import stage


def bank(events, n=4):
    items=[]
    for j,e in enumerate(events):
        kind,a,b,*extra=e
        items.append({"event_type":kind,"onset":a,"offset":b,"event_id":f"{kind}:{j}",
            "measurements":extra[0] if extra else {}})
    return {"events":{"enhanced":items},
        "availability":{"enhanced":{k:"AVAILABLE" for k in set(ROLES)|{"kcomplex"}}},
        "info":{"duration":30*n,"roles":{r:r for r in "fcolrm"}},
        "goods":{r:[True]*n for r in "fcolrm"}}


def n2(events, n=4):
    return bank([("spindle",1,2),*events],n)


def after(d):return d["temporal"]["after"]


def test_late_arousal_keeps_label_but_changes_end_and_next_epoch():
    p=stage(n2([("lamf",30,120),("arousal",55,58)]))
    assert [d["stage"] for d in p]==["N2","N2","N1","N1"]
    assert p[1]["temporal"]["seconds"]=={"N1":2.,"N2":25.,"UNDETERMINED":3.}
    assert after(p[1])["stage"]=="N1"
    assert not after(p[1])["physiological_anchor_valid"]
    assert p[2]["rule_id"]=="N2.4.POST_AROUSAL_MAJORITY"
    assert p[2]["decision_source"]=="CONTINUATION"
    assert after(p[2])["origin_arousal_ids"]==after(p[1])["origin_arousal_ids"]


def test_lamf_only_before_arousal_never_counts_as_post_arousal():
    p=stage(n2([("lamf",30,50),("arousal",55,58),("lamf",60,90)],3))
    assert p[1]["temporal"]["seconds"]["N1"]==0
    assert p[1]["stage"]=="N2" and after(p[1])["stage"]=="UNDETERMINED"
    assert after(p[1])["phase"]=="POST_AROUSAL"
    assert p[2]["stage"]=="N1"


def test_no_majority_fallback_kept_without_resurrecting_anchor():
    p=stage(n2([("lamf",30,60),("arousal",55,58),("lamf",60,70),("lamf",90,120)]))
    assert p[2]["stage"]=="N2"
    assert p[2]["rule_id"]=="FALLBACK.INHERIT_PREVIOUS"
    assert p[2]["previous_context"]["incompatible_evidence"]
    assert after(p[2])["phase"]=="POST_AROUSAL"
    assert not after(p[2])["physiological_anchor_valid"]
    assert p[3]["stage"]=="N1"


def test_fixed_30_second_denominator_not_renormalized():
    p=stage(n2([("lamf",30,90),("arousal",45,48)],3))
    assert p[1]["temporal"]["seconds"]=={"N1":12.,"N2":15.,"UNDETERMINED":3.}
    assert p[1]["rule_id"]=="FALLBACK.INHERIT_PREVIOUS"
    assert after(p[1])["stage"]=="N1"


def test_exact_tie_at_structural_reentry_keeps_fallback_and_end_n2():
    p=stage(n2([("lamf",30,120),("arousal",31,34),("kcomplex",75,76)]))
    assert p[2]["temporal"]["seconds"]=={"N1":15.,"N2":15.,"UNDETERMINED":0.}
    assert p[2]["stage"]=="N1" and p[2]["rule_id"]=="FALLBACK.INHERIT_PREVIOUS"
    assert after(p[2])["stage"]=="N2"
    assert p[3]["stage"]=="N2"


def test_cross_epoch_arousal_preserves_pending_end():
    p=stage(n2([("lamf",30,120),("arousal",58,64)]))
    assert p[1]["stage"]=="N2" and after(p[1])["stage"]=="UNDETERMINED"
    assert p[1]["temporal"]["seconds"]["UNDETERMINED"]==2
    assert p[2]["temporal"]["seconds"]=={"N1":26.,"N2":0.,"UNDETERMINED":4.}
    assert p[2]["stage"]=="N1"


@pytest.mark.parametrize("kind",["kcomplex","spindle"])
def test_post_arousal_k_and_spindle_have_identical_reentry(kind):
    p=stage(n2([("lamf",30,120),("arousal",55,64),(kind,70,71)]))
    assert p[2]["stage"]=="N2"
    assert p[2]["temporal"]["seconds"]["N2"]==20
    assert after(p[2])["physiological_anchor_valid"]
    assert any(t["kind"]=="N2_REENTRY" for t in p[2]["temporal"]["transitions"])


def test_arousal_associated_k_does_not_reenter():
    p=stage(n2([("lamf",30,120),("arousal",55,64),
                ("kcomplex",70,71,{"associated_arousal_ids":["arousal:2"]})]))
    assert p[2]["stage"]=="N1" and not after(p[2])["physiological_anchor_valid"]


@pytest.mark.parametrize("onset,expected",[(62,"N1"),(64,"N2")])
def test_structure_during_arousal_vs_exact_offset(onset,expected):
    p=stage(n2([("lamf",30,120),("arousal",55,64),("spindle",onset,onset+1)]))
    assert p[2]["stage"]==expected


def test_multiple_arousals_invalidate_the_new_anchor_again():
    p=stage(n2([("lamf",30,120),("arousal",31,34),("spindle",40,41),("arousal",55,58)]))
    assert p[1]["stage"]=="N2" and after(p[1])["stage"]=="N1"
    assert not after(p[1])["physiological_anchor_valid"]
    assert after(p[1])["origin_arousal_ids"]==["arousal:4"]
    assert p[2]["stage"]=="N1"


def test_new_n2_onset_before_arousal_cannot_seed_next_epoch():
    p=stage(bank([("vertex",1,2),("lamf",0,90),("spindle",31,32),("arousal",55,58)],3))
    assert [d["stage"] for d in p]==["N1","N2","N1"]
    assert not after(p[1])["physiological_anchor_valid"]


def test_definite_rem_overrides_arousal_fragment_rule_and_end():
    p=stage(n2([("lamf",30,90),("arousal",31,34),("rem",45,46),("low_emg",30,90)],3))
    assert p[1]["rule_id"]=="R.1" and p[1]["stage"]=="R"
    assert after(p[1])["stage"]=="R" and not after(p[1])["repair_active"]
    assert p[2]["stage"]=="R"


def test_existing_rem_termination_priority_is_preserved():
    p=stage(bank([("lamf",0,90),("low_emg",0,90),("rem",1,2),("rem",31,32),("high_emg",30,60)],3))
    assert p[0]["stage"]=="R"
    assert p[1]["rule_id"]=="R.3.EMG_N1" and p[1]["stage"]=="N1"


@pytest.mark.parametrize("kind,expected",[("slow_wave","N3"),("alpha","W")])
def test_higher_stage_clears_old_n2_physiological_context(kind,expected):
    p=stage(n2([("lamf",30,90),("arousal",31,34),(kind,30,50)],3))
    assert p[1]["stage"]==expected and after(p[1])["stage"]==expected
    assert not after(p[1])["repair_active"] and not after(p[1])["physiological_anchor_valid"]


def test_invalid_epoch_clears_temporal_context():
    b=n2([("lamf",30,120),("arousal",55,64)])
    b["goods"]["c"][1]=False
    p=stage(b)
    assert p[1]["stage"]=="UNDETERMINED" and after(p[1])["phase"]=="OTHER"


def test_unknown_lamf_is_not_positive_post_arousal_evidence():
    b=n2([("arousal",31,34)])
    b["availability"]["enhanced"]["lamf"]="UNAVAILABLE"
    p=stage(b)
    assert p[1]["temporal"]["seconds"]["N1"]==0
    assert after(p[1])["phase"]=="POST_AROUSAL"
    assert not after(p[1])["physiological_anchor_valid"]


@pytest.mark.parametrize("onset",[30.,44.999,45.,59.999])
def test_no_arousal_original_n2_onset_window_unchanged(onset):
    b=bank([("vertex",1,2),("lamf",0,120),("spindle",onset,onset+.1)])
    old=enhanced_engine(aggregate(b,"enhanced"))
    new=stage(b)
    assert [(d.stage,d.rule_id) for d in old]==[(d["stage"],d["rule_id"]) for d in new]


def test_no_arousal_continuation_does_not_expand_and_fallback_is_unchanged():
    b=n2([("lamf",30,120)])
    old=enhanced_engine(aggregate(b,"enhanced"))
    new=stage(b)
    keys=("stage","rule_id","decision_source","anchor","chain_length")
    assert [[d.to_dict()[k] for k in keys] for d in old]==[[d[k] for k in keys] for d in new]
    assert new[2]["rule_id"]=="FALLBACK.INHERIT_PREVIOUS"
    from aasm_temporal_r1 import engine as module
    assert module.inherit is inherit


def test_ground_truth_fields_do_not_change_prediction():
    b=n2([("lamf",30,120),("arousal",55,58)])
    before=stage(b)
    b["true_stage"]=["W","R","N3","W"]
    b["annotations"]="forbidden irrelevant payload"
    assert stage(b)==before


def test_frozen_dependency_and_config_hashes_are_identical():
    root=Path(__file__).resolve().parents[1]
    manifest=json.loads((root/"configs/frozen_dependency_manifest.json").read_text())
    for name,expected in manifest.items():
        assert hashlib.sha256((root/name).read_bytes()).hexdigest()==expected


def test_input_bank_is_not_mutated():
    b=n2([("lamf",30,120),("arousal",55,58)])
    original=deepcopy(b)
    stage(b)
    assert b==original


def test_previous_second_half_structure_initializes_next_arousal_context():
    p=stage(bank([("vertex",1,2),("lamf",0,90),
                  ("spindle",25,26),("arousal",31,34)],3))
    assert p[0]["stage"]=="N1"
    assert p[1]["temporal"]["seconds"]=={"N1":26.,"N2":1.,"UNDETERMINED":3.}
    assert p[1]["rule_id"]=="N2.4.POST_AROUSAL_MAJORITY"
    assert not after(p[1])["physiological_anchor_valid"]


def test_previous_window_anchor_cannot_resurrect_after_later_arousal():
    p=stage(n2([("lamf",30,90),("spindle",50,51),("arousal",55,58)],3))
    assert p[1]["stage"]=="N2" and after(p[1])["stage"]=="N1"
    assert p[2]["stage"]=="N1"
    assert not after(p[2])["physiological_anchor_valid"]


def test_rem_to_n2_label_does_not_undo_late_arousal_end_state():
    p=stage(bank([("lamf",0,90),("low_emg",0,30),("rem",1,2),
                  ("spindle",31,32),("arousal",55,58)],3))
    assert p[0]["stage"]=="R"
    assert p[1]["rule_id"]=="R.3.N2" and p[1]["stage"]=="N2"
    assert after(p[1])["stage"]=="N1"
    assert not after(p[1])["physiological_anchor_valid"]
    assert p[2]["rule_id"]=="N2.4.POST_AROUSAL_MAJORITY"


def test_previous_window_gap_prevents_carry_anchor():
    b=bank([("vertex",1,2),("lamf",0,90),("spindle",25,26),("arousal",31,34)],3)
    b["goods"]["f"][0]=False
    p=stage(b)
    assert not any(t["kind"]=="PRIOR_HALF_N2_ANCHOR" for t in p[1]["temporal"]["transitions"])
    assert not p[1]["temporal"]["engaged"]
