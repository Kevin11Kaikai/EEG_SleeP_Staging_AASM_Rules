"""Verify frozen scientific core and optional public-file manifest."""
from _common import ROOT,read_json,check_core
from aasm_v2.provenance import file_hash

if __name__=="__main__":
    print("Core verified",check_core()["source_config_test_hash"])
    manifest=ROOT/"PUBLIC_MANIFEST.json"
    if manifest.exists():
        data=read_json(manifest)
        for name,digest in data["files"].items():assert file_hash(ROOT/name)==digest,name
        print("Public manifest verified",len(data["files"]),"files")
