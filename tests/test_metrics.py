import importlib.util
from pathlib import Path
import numpy as np
from sklearn.metrics import cohen_kappa_score, f1_score
spec=importlib.util.spec_from_file_location("independent_evaluation",Path(__file__).resolve().parents[1]/"scripts/evaluate.py")
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def test_abstention_denominator_and_metrics_match_sklearn():
    y=np.asarray(["W","N1","N2","N3","R","R"])
    p=np.asarray(["W","UNDETERMINED","N2","W","R","UNDETERMINED"])
    result=module.metrics(module.confusion(y,p))
    assert result["accuracy"]==.5
    assert result["coverage"]==4/6
    assert result["conditional_accuracy"]==3/4
    assert abs(result["kappa"]-cohen_kappa_score(y,p,labels=module.STAGES))<1e-12
    assert abs(result["macro_f1"]-f1_score(y,p,labels=module.STAGES[:5],average="macro",zero_division=0))<1e-12


def test_annotation_conflict_partial_and_half_open_boundaries():
    events=[(0,30,"Sleep stage W"),(30,30,"Sleep stage N2"),(60,15,"Sleep stage N1")]
    assert module.align_annotations(events,3).tolist()==["W","N2","UNALIGNED"]
    events.append((35,5,"Sleep stage R"))
    assert module.align_annotations(events,3).tolist()==["W","UNALIGNED","UNALIGNED"]
