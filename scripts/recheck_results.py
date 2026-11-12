"""Recompute aggregate scoring metrics without individual-level data."""
import argparse, csv, json, math
from pathlib import Path

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", default="results/full_v1")
    args=p.parse_args()
    root=Path(__file__).resolve().parents[1]
    folder=root/args.run_dir
    expected=json.loads((folder/"overall_metrics.json").read_text())
    result={}
    for window in ["primary", "full_record"]:
        with (folder/f"confusion_matrix_{window}.csv").open(newline="",encoding="utf-8-sig") as f:
            cm=[[int(r[s]) for s in ["W","N1","N2","N3","R","UNDETERMINED"]] for r in csv.DictReader(f)]
        n=sum(map(sum,cm)); rows=list(map(sum,cm)); cols=[sum(r[j] for r in cm) for j in range(6)]
        acc=sum(cm[i][i] for i in range(5))/n
        chance=sum(rows[i]*cols[i] for i in range(5))/(n*n)
        kappa=(acc-chance)/(1-chance)
        f1=sum(2*cm[i][i]/(rows[i]+cols[i]) if rows[i]+cols[i] else 0 for i in range(5))/5
        values={"epochs":n,"accuracy":acc,"kappa":kappa,"macro_f1":f1}
        for key,value in values.items():
            assert math.isclose(value,expected[window][key],abs_tol=1e-12), (window,key)
        result[window]=values
    print(json.dumps({"status":"VERIFIED","scope":"151-record aggregate matrices only","metrics":result},indent=2))

if __name__=="__main__": main()
