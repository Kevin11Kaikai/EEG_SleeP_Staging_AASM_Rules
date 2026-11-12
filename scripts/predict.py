"""Run frozen V0–V4 and Temporal R1 on a local PSG without scoring labels."""
import argparse
from datetime import datetime,timezone
import gzip
import json
from pathlib import Path
import sys
from _common import ROOT,read_json,check_core
from aasm_v2.pipeline import predict,gzip_json
from aasm_v2.provenance import file_hash,write_json
from aasm_temporal_r1 import stage


def guard(event,args):
    if event=="open" and isinstance(args[0],str):
        p=args[0].lower()
        if "sleepscoring" in p or "aligned_evaluation" in p:
            raise RuntimeError("Prediction may not read reference annotations")


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--psg",type=Path,nargs="+",required=True)
    parser.add_argument("--run-dir",type=Path,required=True)
    args=parser.parse_args();freeze=check_core()
    sys.addaudithook(guard)
    cfg=read_json(ROOT/"configs/protocol.json")
    for psg in args.psg:
        dest=args.run_dir/"records"/psg.stem
        frozen=predict(psg,dest,cfg,ROOT)
        with gzip.open(dest/"events.json.gz","rt",encoding="utf8") as handle:bank=json.load(handle)
        target=dest/"temporal_r1_predictions.json.gz"
        marker=dest/"r1_checkpoint.json"
        identity={"core":freeze["source_config_test_hash"],"events":file_hash(dest/"events.json.gz")}
        if marker.exists():
            old=read_json(marker)
            assert old["identity"]==identity and old["prediction_sha256"]==file_hash(target)
        else:
            gzip_json(target,stage(bank))
            write_json(marker,{"status":"SUCCESS","identity":identity,
                "prediction_sha256":file_hash(target),"created_utc":datetime.now(timezone.utc).isoformat()})
        print(psg.stem,"frozen variants + R1 complete; labels not accessed",flush=True)


if __name__=="__main__":main()
