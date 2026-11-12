"""Supplemental integration checks of actual event association after code freeze.

Source/config stay unchanged. Only upstream candidate timing is controlled here.
"""
import json
from pathlib import Path
import numpy as np
import pytest
import aasm_v2.events as module

CFG=json.loads((Path(__file__).resolve().parents[1]/"configs/protocol.json").read_text())


@pytest.mark.parametrize("onset,associated",[(20.5,True),(22.,True),(22.001,False)])
def test_k_arousal_concurrent_and_one_second_inclusive_boundary(monkeypatch,onset,associated):
    sf=128;t=np.arange(30*sf)/sf
    signal=10*np.sin(2*np.pi*5*t)+.1*np.sin(2*np.pi*17*t)
    info={"record_id":"SYNTHETIC","duration":30,"roles":{r:r for r in "fcolrm"},
          "channels":[{"name":r,"min_uv":-800,"max_uv":800} for r in "fcolrm"]}
    monkeypatch.setattr(module,"header",lambda p:info)
    monkeypatch.setattr(module,"read_channel",lambda p,r,i:(signal.copy(),sf))
    monkeypatch.setattr(module,"yasa_candidates",lambda *a,**k:([],[]))
    monkeypatch.setattr(module,"biphasic_candidates",lambda *a,**k:[(20.,21.,{})])
    monkeypatch.setattr(module,"enhanced_arousal",lambda *a,**k:([(onset,onset+3,{})],{"accepted":1}))
    result=module.detect(Path("synthetic.edf"),CFG)
    k=next(e for e in result["events"]["enhanced"] if e["event_type"]=="kcomplex")
    assert bool(k["measurements"]["associated_arousal_ids"])==associated
