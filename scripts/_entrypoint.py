"""Portable command wrappers; no scientific algorithm changes."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

def run(command):
    from hmc_sbtb.cli import main
    sys.argv.insert(1, command)
    main()
