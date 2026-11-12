import os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
os.environ["MNE_USE_NUMBA"]="false"
os.environ["NUMBA_CACHE_DIR"]=str(ROOT/".runtime/numba")
os.environ["MPLCONFIGDIR"]=str(ROOT/".runtime/matplotlib")
