import inspect
from pathlib import Path
import pytest
from aasm_v2.types import Truth as T, Evidence, EpochEvidence, StageDecision, Event
from aasm_v2.temporal import duration, intersect, runs
from aasm_v2.rules import original_engine, enhanced_engine, resolve_mbm
from aasm_v2.rules.registry import registry
from aasm_v2.data import unit_factor


def epoch(i=0, **values):
    keys = ["slow_wave", "alpha", "blink", "irregular_rem_normal", "lamf", "rem",
            "low_emg", "high_emg", "normal_emg", "rem_low_overlap", "spindle", "kcomplex",
            "arousal", "n2_spindle", "n2_kcomplex", "first_half_n2", "arousal_then_sem",
            "slowing", "vertex", "sem", "mbm"]
    # These injected logical fixtures have KNOWN negatives; they are not real PSG.
    ev = {k: Evidence(values.get(k, 0), [f"{i}:{k}"] if values.get(k, 0) else [],
                       negative_validated=True) for k in keys}
    return EpochEvidence(i, True, ev, {"alpha_generator": False, "n2_event_in_epoch": False})


@pytest.mark.parametrize("a,b,result", [(T.TRUE,T.UNKNOWN,T.UNKNOWN),(T.FALSE,T.UNKNOWN,T.FALSE),
                                        (T.TRUE,T.TRUE,T.TRUE),(T.UNKNOWN,T.UNKNOWN,T.UNKNOWN)])
def test_three_state_and(a,b,result):
    assert a & b is result


def test_no_implicit_boolean_or_unknown_negative():
    with pytest.raises(TypeError):
        bool(T.UNKNOWN)
    assert Evidence(0, detector_status="PARTIALLY_SUPPORTED").present() is T.UNKNOWN
    assert Evidence(None).threshold(0) is T.UNKNOWN
    assert ~T.UNKNOWN is T.UNKNOWN
    assert T.TRUE | T.UNKNOWN is T.TRUE


@pytest.mark.parametrize("seconds,stage", [(5.999,"UNDETERMINED"),(6,"N3"),(6.001,"N3")])
def test_n3_twenty_percent_boundary(seconds,stage):
    assert enhanced_engine([epoch(slow_wave=seconds)], "ABSTAIN")[0].stage == stage


@pytest.mark.parametrize("key", ["alpha", "blink", "irregular_rem_normal"])
def test_independent_wake_majority(key):
    assert enhanced_engine([epoch(**{key:15})], "ABSTAIN")[0].stage != "W"
    d = enhanced_engine([epoch(**{key:15.01})], "ABSTAIN")[0]
    assert d.stage == "W"
    assert enhanced_engine([epoch(alpha=10,blink=10)], "ABSTAIN")[0].stage != "W"


def test_n3_priority_over_wake():
    assert enhanced_engine([epoch(slow_wave=6,alpha=20)])[0].stage == "N3"


def test_n1_nonalpha_paths_and_alpha_attenuation():
    for key in ("slowing", "vertex", "sem"):
        assert enhanced_engine([epoch(**{key:1})])[0].rule_id == "N1.2"
    a, b = epoch(alpha=20), epoch(1,lamf=20,alpha=2)
    a.attributes["alpha_generator"] = b.attributes["alpha_generator"] = True
    assert enhanced_engine([a,b])[1].rule_id == "N1.1"


def test_rem_onset_requires_temporal_overlap():
    e = epoch(lamf=20,rem=1,low_emg=20)
    assert enhanced_engine([e],"ABSTAIN")[0].stage != "R"
    e.events["rem_low_overlap"] = Evidence(.1,negative_validated=True)
    assert enhanced_engine([e],"ABSTAIN")[0].rule_id == "R.1"


def test_rem_termination_has_distinct_targets():
    r = epoch(lamf=20,rem=1,low_emg=20,rem_low_overlap=.1)
    for end, stage in ((epoch(1,alpha=20),"W"),(epoch(1,slow_wave=10),"N3"),
                       (epoch(1,high_emg=20,lamf=20),"N1"),
                       (epoch(1,arousal_then_sem=1),"N1"),(epoch(1,first_half_n2=1),"N2")):
        assert enhanced_engine([r,end],"ABSTAIN")[1].stage == stage


def test_n2_from_n3_and_arousal_reset():
    assert enhanced_engine([epoch(slow_wave=8),epoch(1)],"ABSTAIN")[1].rule_id == "N2.3"
    start = epoch(n2_spindle=1)
    ar = epoch(1,arousal=4,lamf=20)
    after = epoch(2,lamf=20)
    reentry = epoch(3,n2_spindle=1)
    reentry.attributes["n2_after_arousal_ids"] = ["new-spindle"]
    out = enhanced_engine([start,ar,after,reentry],"ABSTAIN")
    assert out[1].rule_id == "N2.4.AROUSAL"
    assert out[2].stage == "UNDETERMINED"
    assert out[3].rule_id == "N2.1"


def test_unknown_arousal_does_not_certify_continuation():
    a,b = epoch(slow_wave=8),epoch(1)
    b.events["arousal"] = Evidence(0,detector_status="PARTIALLY_SUPPORTED")
    assert enhanced_engine([a,b],"ABSTAIN",negative_policy="clinical_strict")[1].stage == "UNDETERMINED"


def test_mbm_consecutive_lookahead_tail_and_unknown():
    rows = [epoch(mbm=20),epoch(1,mbm=20),epoch(2)]
    ds = [StageDecision(0),StageDecision(1),StageDecision(2,stage="N2")]
    resolve_mbm(ds,rows)
    assert [d.stage for d in ds] == ["N2"]*3
    assert ds[0].rule_id == "MBM.2"
    tail = [StageDecision(0)]
    resolve_mbm(tail,[epoch(mbm=20)])
    assert tail[0].stage == "UNDETERMINED"
    ds = [StageDecision(0,stage="W"),StageDecision(1),StageDecision(2)]
    resolve_mbm(ds,[epoch(),epoch(1,mbm=20),epoch(2)])
    assert ds[1].rule_id == "MBM.1"


def test_inheritance_chain_and_quality_barrier():
    rows = [epoch(alpha=20)] + [epoch(i) for i in range(1,6)]
    out = enhanced_engine(rows,"CONDITIONAL_INHERIT",2)
    assert [x.stage for x in out] == ["W","W","W","UNDETERMINED","UNDETERMINED","UNDETERMINED"]
    out = enhanced_engine(rows,"INHERIT_PREVIOUS")
    assert out[-1].chain_length == 5 and out[-1].anchor == 0
    rows[2].signals_good = False
    assert enhanced_engine(rows)[2].stage == "UNDETERMINED"


def test_intervals_cross_epochs():
    assert duration([(29,31),(30,33)],0,30) == 1
    assert duration([(29,31),(30,33)],30,60) == 3
    assert intersect([(1,2)],[(2,3)]) == []


@pytest.mark.parametrize("unit,factor", [("uV",1),("µV",1),("mV",1000),("V",1e6)])
def test_physical_units(unit,factor):
    assert unit_factor(unit) == factor


def test_prediction_module_has_no_annotation_imports():
    package = Path(__file__).resolve().parents[1] / "src/aasm_v2"
    for p in package.rglob("*.py"):
        if "evaluation" in p.parts:
            continue
        import ast
        tree = ast.parse(p.read_text(encoding="utf8"))
        for node in ast.walk(tree):
            if isinstance(node,ast.ImportFrom):
                assert not any(x in (node.module or "") for x in ("evaluation","annotations","hmc_sbtb"))
    assert "label" not in inspect.signature(enhanced_engine).parameters


def test_registry_contains_every_rule():
    expected = {"W.1","W.2","W.3","N1.1","N1.2","N2.1","N2.2","N2.3","N2.4",
                "N3.1","R.1","R.2","R.3","MBM.1","MBM.2"}
    assert expected <= {x["rule_id"] for x in registry()}


def test_rem_termination_overrides_simultaneous_onset():
    a = epoch(lamf=20,rem=1,low_emg=20,rem_low_overlap=.1)
    b = epoch(1,lamf=20,rem=1,low_emg=20,rem_low_overlap=.1,arousal_then_sem=1)
    assert enhanced_engine([a,b],"ABSTAIN")[1].rule_id == "R.3.AROUSAL_SEM"


def test_mbm_cannot_seed_its_future_resolution():
    rows = [epoch(mbm=20,slow_wave=10),epoch(1)]
    assert all(d.stage == "UNDETERMINED" for d in enhanced_engine(rows))


def test_missing_alpha_comparison_stays_unknown():
    a,b = epoch(),epoch(1,lamf=20)
    a.events["alpha"] = Evidence(None)
    a.attributes["alpha_generator"] = b.attributes["alpha_generator"] = True
    result=enhanced_engine([a,b],"ABSTAIN")[1]
    assert next(x["value"] for x in result.trace if x["rule_id"]=="N1.1") == "UNKNOWN"


def test_n2_reentry_accepts_previous_second_half_after_arousal():
    a,b,c = epoch(n2_spindle=1),epoch(1,arousal=3,lamf=20),epoch(2,n2_spindle=1)
    b.attributes["last_arousal_end"] = 40
    c.attributes["n2_window_onsets"] = [["spindle-at-50",50]]
    assert enhanced_engine([a,b,c],"ABSTAIN")[2].rule_id == "N2.1"


def test_n2_and_rem_continuation_known_operational_negatives():
    a,b = epoch(n2_spindle=1),epoch(1,lamf=20)
    a.attributes["n2_event_in_epoch"] = True
    assert enhanced_engine([a,b],"ABSTAIN")[1].rule_id == "N2.2"
    r=epoch(lamf=20,rem=1,low_emg=20,rem_low_overlap=.1)
    assert enhanced_engine([r,epoch(1,lamf=20,low_emg=20)],"ABSTAIN")[1].rule_id == "R.2"
