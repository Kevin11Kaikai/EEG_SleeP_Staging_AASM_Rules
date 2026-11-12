"""Display frozen aggregate ablations; prediction CLI computes all variants."""
from _common import ROOT,read_json

if __name__=="__main__":
    data=read_json(ROOT/"results/frozen_v2/metrics.json")["primary_all"]
    for v in ("Legacy","V0","V1","V2","V3","V4"):
        m=data[v]
        print(f'{v:8} Accuracy={m["accuracy"]:.6f} Macro-F1={m["macro_f1"]:.6f} kappa={m["kappa"]:.6f} Coverage={m["coverage"]:.6f}')
    print("For new PSGs, predict.py computes all frozen variants in one event pass.")
