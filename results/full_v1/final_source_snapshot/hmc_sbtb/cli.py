"""Separate prediction, evaluation, and audit commands with lazy imports."""
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(prog="hmc_sbtb")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ["discover", "download", "predict", "evaluate", "ablation", "audit"]:
        p = sub.add_parser(name)
        p.add_argument("--data-root", default="data")
        p.add_argument("--subset", choices=["smoke", "pilot", "all"], default="smoke")
        p.add_argument("--config", default="configs/paper_inspired.yaml")
        p.add_argument("--run-dir", default="results/smoke_v2")
        p.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if args.command == "discover":
        import json
        root = Path("data" if args.data_root == "auto" else args.data_root)
        print(json.dumps({"data_root": str(root.resolve()), "psg_count": len(list((root / "recordings").glob("SN???.edf"))),
                          "official_metadata_present": (root / "RECORDS").exists()}, indent=2))
    elif args.command == "download":
        from .download_hmc import download
        download(args.data_root, args.subset, args.workers)
    elif args.command == "predict":
        from .pipeline import predict
        predict(args.config, args.subset, args.data_root, args.run_dir)
    elif args.command == "evaluate":
        from .evaluation import evaluate
        evaluate(Path(args.run_dir))
    elif args.command == "ablation":
        from .ablation import ablate
        ablate(Path(args.run_dir))
    elif args.command == "audit":
        from .visualization import audit
        audit(Path(args.run_dir))


if __name__ == "__main__":
    main()
