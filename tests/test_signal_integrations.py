import json
from pathlib import Path
import numpy as np
import pytest
from aasm_v2.events import yasa_candidates,detect
from aasm_v2.evidence import aggregate

CFG=json.loads((Path(__file__).resolve().parents[1]/"configs/protocol.json").read_text())


@pytest.mark.parametrize("kind",["slow_wave","spindle","rem"])
def test_yasa_physiological_positive_signals(kind):
    sf=128;t=np.arange(60*sf)/sf;rng=np.random.default_rng(100)
    background=rng.normal(0,1,len(t))+4*np.sin(2*np.pi*3*t)
    if kind=="slow_wave":
        x=80*np.sin(2*np.pi*.8*t)+background
        inputs=[x]
    elif kind=="spindle":
        x=background.copy()
        for center in (10,20,30,40,50):
            x+=40*np.sin(2*np.pi*13*t)*np.exp(-((t-center)/.45)**2)
        inputs=[x]
    else:
        x=background.copy()
        for center in (10,20,30,40,50):
            x+=120*np.exp(-((t-center)/.18)**2)
        inputs=[x,-x]
    events,log=yasa_candidates(inputs,sf,np.ones(2,bool),kind,CFG["yasa"])
    assert events
    assert all(entry["status"]=="EVENTS_RETURNED" for entry in log)


def test_yasa_failed_amplitude_is_not_no_event():
    sf=128;rng=np.random.default_rng(20261112)
    x=rng.normal(0,.12,sf*60)
    events,log=yasa_candidates([x,-x],sf,np.ones(2,bool),"rem",CFG["yasa"])
    assert not events and log[0]["status"]=="INVALID_SIGNAL"


def test_movement_requires_multimodal_obscuration_and_tone_is_measured(monkeypatch):
    import aasm_v2.events as module
    sf=256;t=np.arange(90*sf)/sf;rng=np.random.default_rng(12)
    channels={r:10*np.sin(2*np.pi*5*t)+rng.normal(0,.5,len(t)) for r in "fcolr"}
    channels["m"]=2*np.sin(2*np.pi*20*t)+rng.normal(0,.1,len(t))
    movement=(t>=5)&(t<22)
    for role in "fc":
        channels[role][movement]+=250*np.sin(2*np.pi*20*t[movement])
    channels["m"][movement]+=100*np.sin(2*np.pi*25*t[movement])
    info={"record_id":"SYNTHETIC","duration":90,"roles":{r:r for r in "fcolrm"},
          "channels":[{"name":r,"min_uv":-800,"max_uv":800} for r in "fcolrm"]}
    monkeypatch.setattr(module,"header",lambda path:info)
    monkeypatch.setattr(module,"read_channel",lambda path,role,info:(channels[role].copy(),sf))
    bank=detect(Path("synthetic.edf"),CFG)
    rows=aggregate(bank,"enhanced")
    assert rows[0].get("mbm").seconds>15
    assert rows[1].get("mbm").seconds==0
    assert bank["diagnostics"]["backgrounds"]["emg_floor_uv"]>0
    assert rows[1].get("low_emg").seconds>15
