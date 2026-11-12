import json
from pathlib import Path
import numpy as np
import pytest
from aasm_v2.events.arousal import legacy_arousal, enhanced_arousal
from aasm_v2.events.morphology import biphasic_candidates, ocular_candidates, vertex_candidates
from aasm_v2.preprocessing import spectra, prepare, quality
from aasm_v2.events import measured_windows
from aasm_v2.temporal import duration, runs
from aasm_v2.evidence import aggregate

CFG = json.loads((Path(__file__).resolve().parents[1] / "configs/protocol.json").read_text())


def synthetic_spec(start=12, length=5, total=40):
    out = {"theta":np.full(total,.2),"alpha":np.full(total,.1),"beta":np.full(total,.05),
           "frequency":np.full(total,3.),"total":np.full(total,100.),"valid":np.ones(total,bool)}
    out["alpha"][start:start+length] = .65
    out["frequency"][start:start+length] = 10
    return out


def arouse(spec, sleep=None, sp=None, art=None, rem=None, emg=None, emg_valid=None):
    n=len(spec["valid"])
    return enhanced_arousal(spec, np.ones(n,bool) if sleep is None else sleep,
        np.zeros(n,bool) if sp is None else sp, np.zeros(n,bool) if art is None else art,
        np.zeros(n,bool) if rem is None else rem, np.ones(n) if emg is None else emg,
        np.ones(n,bool) if emg_valid is None else emg_valid, CFG["enhanced"]["arousal"])


def test_fixed_reference_detects_sustained_shift_legacy_self_suppresses():
    old=np.r_[np.full(12,.1),np.full(5,.5),np.full(23,.1)]
    ev,cnt=legacy_arousal(old,np.ones(40,bool))
    assert ev == [] and cnt["one_second_candidates"] > 0
    ev,cnt=arouse(synthetic_spec())
    assert [(a,b) for a,b,_ in ev] == [(12.,17.)]
    assert ev[0][2]["reference_end"] == 12


@pytest.mark.parametrize("start,length,expected", [(9,4,0),(10,3,1),(10,2,0),(12,5,1)])
def test_ten_plus_three_boundary(start,length,expected):
    assert len(arouse(synthetic_spec(start,length))[0]) == expected


def test_rem_emg_condition():
    spec=synthetic_spec()
    rem=np.ones(40,bool)
    assert arouse(spec,rem=rem)[0] == []
    emg=np.ones(40);emg[13]=3
    ev,cnt=arouse(spec,rem=rem,emg=emg)
    assert len(ev)==1 and ev[0][2]["emg_increase_seconds"] == 1
    valid=np.ones(40,bool);valid[13]=False
    assert arouse(spec,rem=rem,emg=emg,emg_valid=valid)[0] == []


@pytest.mark.parametrize("reason", ["spindle","artifact","invalid","not_sleep"])
def test_arousal_rejects_counterexamples(reason):
    spec=synthetic_spec(); mask=np.zeros(40,bool);mask[12:17]=True
    if reason=="spindle":
        assert arouse(spec,sp=mask)[0] == []
    elif reason=="artifact":
        assert arouse(spec,art=mask)[0] == []
    elif reason=="invalid":
        spec["valid"][8]=False
        assert arouse(spec)[0] == []
    else:
        assert arouse(spec,sleep=np.zeros(40,bool))[0] == []


def test_real_waveform_frequency_change_and_lamf_zero_crossing():
    sf=128;t=np.arange(60*sf)/sf
    x=20*np.sin(2*np.pi*3*t)
    use=(t>=20)&(t<25); x[use]=20*np.sin(2*np.pi*10*t[use])
    spec=spectra(x,sf,np.ones(2,bool))
    ev,cnt=arouse(spec)
    assert any(a<=20 and b>=25 for a,b,_ in ev)
    theta=15*np.sin(2*np.pi*5*t)
    threshold=np.mean(np.abs(theta))-.01*np.std(theta)
    assert runs(np.abs(theta)<threshold,sf,1) == []
    lw=spectra(theta,sf,np.ones(2,bool),1,.5)
    segments=measured_windows(lw,(lw["rms"]<=20)&(lw["theta"]>=.01))
    assert duration(segments,0,60) == 60


def test_invalid_segments_cannot_contaminate_resampling():
    sf=256;t=np.arange(90*sf)/sf
    x=10*np.sin(2*np.pi*6*t);y=x.copy()
    y[30*sf:60*sf]=1e8
    good=np.array([True,False,True])
    a=prepare(x,sf,128,good,[.3,35]);b=prepare(y,sf,128,good,[.3,35])
    np.testing.assert_array_equal(a,b)


def test_kcomplex_shape_boundaries_and_noise():
    sf=128;t=np.arange(30*sf)/sf
    x=-100*np.exp(-((t-10)/.12)**2)+60*np.exp(-((t-10.5)/.22)**2)
    events=biphasic_candidates(x,sf,np.ones(1,bool),CFG["enhanced"]["kcomplex"])
    assert any(a<10<b and b-a>=.5 for a,b,_ in events)
    assert biphasic_candidates(np.zeros_like(x),sf,np.ones(1,bool),CFG["enhanced"]["kcomplex"]) == []


def test_dual_eog_polarity_separates_blink_and_sem():
    sf=128;t=np.arange(30*sf)/sf
    blink=120*np.exp(-((t-10)/.10)**2)
    found=ocular_candidates(blink,blink,sf,np.ones(1,bool),CFG["enhanced"])
    assert found["blink"] and not found["sem"]
    slow=80*np.sin(2*np.pi*.35*t)*np.exp(-((t-15)/3)**2)
    found=ocular_candidates(slow,-slow,sf,np.ones(1,bool),CFG["enhanced"])
    assert found["sem"] and not found["blink"]


def test_vertex_candidate_and_artifact_negative():
    sf=128;t=np.arange(30*sf)/sf
    x=-100*np.exp(-((t-10)/.08)**2)
    found=vertex_candidates(x,x*.2,sf,np.ones(1,bool),CFG["enhanced"]["vertex"])
    assert found
    assert not vertex_candidates(x,x*2,sf,np.ones(1,bool),CFG["enhanced"]["vertex"])


def make_bank(onsets):
    events=[{"event_type":"spindle","onset":a,"offset":a+1,"event_id":str(a),"measurements":{}} for a in onsets]
    return {"info":{"duration":90,"roles":{r:r for r in "fcolm r"}},
            "events":{"original":events},"availability":{"original":{"spindle":"AVAILABLE","alpha":"AVAILABLE"}},
            "goods":{r:[True]*3 for r in "fcolrm"}}


@pytest.mark.parametrize("onset,included", [(14.999,False),(15,True),(44.999,True),(45,False)])
def test_n2_fifteen_second_window(onset,included):
    row=aggregate(make_bank([onset]),"original")[1]
    assert bool(row.get("n2_spindle").seconds) == included


def test_missing_eog_evidence_unknown():
    row=aggregate(make_bank([]),"original")[0]
    assert row.get("rem").seconds is None


def test_unknown_rem_context_does_not_bypass_chin_condition():
    assert arouse(synthetic_spec(),rem=np.full(40,-1))[0] == []


def test_arousal_then_sem_crosses_epoch_boundary():
    bank=make_bank([])
    bank["events"]["original"]=[
        {"event_type":"arousal","onset":26,"offset":29,"event_id":"ar","measurements":{}},
        {"event_type":"sem","onset":31,"offset":33,"event_id":"sem","measurements":{}}]
    bank["availability"]["original"].update(arousal="PARTIALLY_SUPPORTED",sem="PARTIALLY_SUPPORTED")
    assert aggregate(bank,"original")[1].get("arousal_then_sem").seconds == 1
