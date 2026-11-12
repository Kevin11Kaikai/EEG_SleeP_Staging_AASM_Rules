"""Recompute public aggregate scores from saved confusion matrices only."""
import json
from pathlib import Path
import numpy as np
from _common import ROOT,read_json


def recompute(cm):
    cm=np.asarray(cm);n=cm.sum();diag=cm.diagonal();row=cm.sum(1);col=cm.sum(0)
    f1=np.divide(2*diag,row+col,out=np.zeros(6),where=(row+col)>0)
    pe=(row@col)/n**2
    return {"accuracy":float(diag.sum()/n),"macro_f1":float(f1[:5].mean()),
            "kappa":float((diag.sum()/n-pe)/(1-pe))}


def main():
    count=0
    for relative in ("results/frozen_v2/metrics.json","results/temporal_r1_development/metrics.json"):
        for scope,models in read_json(ROOT/relative).items():
            for name,expected in models.items():
                got=recompute(expected["confusion"])
                assert all(abs(got[k]-expected[k])<1e-12 for k in got),(scope,name)
                count+=1
    print("Verified",count,"aggregate model/scope score sets; no clinical rows read")


if __name__=="__main__":main()
