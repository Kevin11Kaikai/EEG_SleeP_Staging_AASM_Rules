"""Evaluate newly saved local predictions against local annotations only."""
import argparse
import csv
import gzip
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from _common import ROOT,read_json,check_core
import numpy as np
from aasm_v2.provenance import file_hash,write_json

STAGES=["W","N1","N2","N3","R","UNDETERMINED"]
MAPPING={"Sleep stage "+s:s for s in STAGES[:5]}
MAPPING.update({"Sleep stage 1":"N1","Sleep stage 2":"N2","Sleep stage 3":"N3"})


def confusion(y,p):
    cm=np.zeros((6,6),dtype=np.int64);lookup={s:i for i,s in enumerate(STAGES)}
    np.add.at(cm,([lookup[s] for s in y],[lookup.get(s,5) for s in p]),1)
    return cm


def metrics(cm):
    n=int(cm.sum());correct=int(np.trace(cm));pred=cm.sum(0);support=cm.sum(1)
    precision=np.divide(cm.diagonal(),pred,out=np.zeros(6),where=pred>0)
    recall=np.divide(cm.diagonal(),support,out=np.zeros(6),where=support>0)
    f1=np.divide(2*precision*recall,precision+recall,out=np.zeros(6),where=precision+recall>0)
    if not n:raise ValueError("No evaluable epochs")
    pe=float(pred@support)/n**2;accepted=n-int(pred[5]);ca=correct/accepted if accepted else None
    return {"n":n,"correct":correct,"accuracy":correct/n,"macro_f1":float(f1[:5].mean()),
       "kappa":(correct/n-pe)/(1-pe) if pe<1 else None,"coverage":accepted/n,
       "abstention_rate":int(pred[5])/n,"conditional_accuracy":ca,
       "accepted_risk":1-ca if ca is not None else None,
       "per_stage":{s:{"precision":float(precision[i]),"recall":float(recall[i]),
             "f1":float(f1[i]),"support":int(support[i])} for i,s in enumerate(STAGES[:5])},
       "confusion":cm.tolist()}


def number(value):
    return sum(float(v)*60**i for i,v in enumerate(reversed(value.strip().split(":"))))


def align_annotations(events,n):
    labels=[set() for _ in range(n)];seconds=np.zeros(n);bad=np.zeros(n,bool)
    for a,d,s in events:
        if d<=0:continue
        for i in range(max(0,int(a//30)),min(n,int(np.ceil((a+d)/30)))):
            overlap=max(0,min((i+1)*30,a+d)-max(i*30,a))
            if not overlap:continue
            if s in MAPPING:seconds[i]+=overlap;labels[i].add(MAPPING[s])
            elif "sleep stage" in s.lower():bad[i]=True
    return np.asarray([next(iter(v)) if len(v)==1 and abs(seconds[i]-30)<1e-6 and not bad[i]
                       else "UNALIGNED" for i,v in enumerate(labels)])


def reference(data,rid,n):
    import mne
    with (data/(rid+"_sleepscoring.txt")).open(encoding="utf-8-sig") as f:
        events=[(number(r["Recording onset"]),number(r["Duration"]),r["Annotation"].strip())
                for r in csv.DictReader(f,skipinitialspace=True)]
    txt=align_annotations(events,n)
    raw=mne.io.read_raw_edf(str(data/(rid+".edf")),preload=False,verbose="ERROR")
    start=raw.info["meas_date"];raw.close()
    ann=mne.read_annotations(str(data/(rid+"_sleepscoring.edf")))
    offset=(ann.orig_time-start).total_seconds() if ann.orig_time and start else 0
    edf=align_annotations(zip(ann.onset+offset,ann.duration,[s.strip() for s in ann.description]),n)
    full=(txt==edf)&np.isin(txt,STAGES[:5]);primary=full.copy()
    nonwake=np.flatnonzero(full&(txt!="W"))
    if len(nonwake):primary[:nonwake[0]]=False;primary[nonwake[-1]+1:]=False
    else:primary[:]=False
    return txt,{"primary":primary,"full":full}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--data-dir",type=Path,required=True,help="Directory with local PSG and scoring EDF/TXT files")
    parser.add_argument("--run-dir",type=Path,required=True)
    args=parser.parse_args();check_core()
    matrices={scope:{} for scope in ("primary","full")};records=[]
    for dest in sorted((args.run_dir/"records").iterdir()):
        if not dest.is_dir():continue
        marker=read_json(dest/"checkpoint.json");r1=read_json(dest/"r1_checkpoint.json")
        assert marker["status"]==r1["status"]=="SUCCESS"
        for name,digest in marker["artifacts"].items():assert file_hash(dest/name)==digest,name
        assert file_hash(dest/"temporal_r1_predictions.json.gz")==r1["prediction_sha256"]
        with gzip.open(dest/"predictions.json.gz","rt",encoding="utf8") as f:preds=json.load(f)
        with gzip.open(dest/"temporal_r1_predictions.json.gz","rt",encoding="utf8") as f:preds["R1"]=json.load(f)
        y,masks=reference(args.data_dir,dest.name,len(preds["R1"]))
        records.append(dest.name)
        for model,ds in preds.items():
            p=np.asarray([d["stage"] for d in ds])
            for scope,mask in masks.items():
                matrices[scope].setdefault(model,[]).append(confusion(y[mask],p[mask]))
    if not records:raise ValueError("No completed local prediction records")
    summary={scope:{m:metrics(np.asarray(cms).sum(0)) for m,cms in variants.items()}
             for scope,variants in matrices.items()}
    write_json(args.run_dir/"evaluation/metrics.json",summary)
    write_json(args.run_dir/"evaluation/local_manifest.json",{"records":records,
        "protocol":"complete TXT/EDF-agreed 30s; first-to-last reliable non-Wake primary",
        "evaluator_sha256":file_hash(Path(__file__))})
    print("Evaluated",len(records),"local records; do not publish local clinical artifacts")


if __name__=="__main__":main()
