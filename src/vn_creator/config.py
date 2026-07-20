"""Central paths and settings for the VN scene pipeline.

Large artifacts (model weights, generated audio/video) live on /data
because the home partition has almost no free space.
"""
import os
from pathlib import Path

# This machine is shared; GPU 0 (and often 3) run other tenants' jobs.
# Pin everything to a GPU that's actually free unless the caller overrides it.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")

DATA_ROOT = Path(os.environ.get("VN_DATA_ROOT", "/data/shasegawa/vn"))
MODELS_DIR = DATA_ROOT / "models"
CACHE_DIR = DATA_ROOT / "cache"
OUTPUTS_DIR = DATA_ROOT / "outputs"
TMP_DIR = DATA_ROOT / "tmp"

REPO_ROOT = Path(__file__).resolve().parents[2]
ASSETS_DIR = REPO_ROOT / "data" / "samples"

for d in (MODELS_DIR, CACHE_DIR, OUTPUTS_DIR, TMP_DIR):
    d.mkdir(parents=True, exist_ok=True)

# Keep HF/torch caches off the home partition.
os.environ.setdefault("HF_HOME", str(CACHE_DIR / "huggingface"))
os.environ.setdefault("HF_HUB_CACHE", str(CACHE_DIR / "huggingface" / "hub"))
os.environ.setdefault("TORCH_HOME", str(CACHE_DIR / "torch"))

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
ANTHROPIC_MODEL = os.environ.get("VN_DIRECTOR_MODEL", "claude-sonnet-5")
