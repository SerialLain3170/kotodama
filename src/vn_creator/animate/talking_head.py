"""Talking-head animation via SadTalker: character still + audio -> video clip.

Invoked as a subprocess (not imported) because SadTalker's inference.py relies
on being run from its own repo root (relative `src.*` imports, relative
`./checkpoints` paths) and on argparse globals.
"""
import shutil
import subprocess
from pathlib import Path

from vn_creator import config

SADTALKER_DIR = Path("/data/shasegawa/vn/third_party/SadTalker")
VENV_PYTHON = Path("/data/shasegawa/vn/venv/bin/python")


def animate(
    character_image: str | Path,
    audio_path: str | Path,
    out_path: str | Path = None,
    enhance: bool = True,
) -> Path:
    """Run SadTalker on a single still + audio clip, return the final mp4 path."""
    character_image = Path(character_image).resolve()
    audio_path = Path(audio_path).resolve()
    out_path = Path(out_path) if out_path else config.TMP_DIR / "talking_head.mp4"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    result_dir = config.TMP_DIR / "sadtalker_run"
    if result_dir.exists():
        shutil.rmtree(result_dir)
    result_dir.mkdir(parents=True)

    cmd = [
        str(VENV_PYTHON), "inference.py",
        "--driven_audio", str(audio_path),
        "--source_image", str(character_image),
        "--result_dir", str(result_dir),
        "--still",
        "--preprocess", "full",
    ]
    if enhance:
        cmd += ["--enhancer", "gfpgan"]

    subprocess.run(cmd, cwd=SADTALKER_DIR, check=True)

    produced = sorted(result_dir.glob("*.mp4"))
    if not produced:
        raise RuntimeError(f"SadTalker produced no mp4 in {result_dir}")
    # SadTalker's top-level output is the one named exactly after the run timestamp
    # (no ## / _full / _enhanced suffix in its stem).
    final = max(produced, key=lambda p: p.stat().st_mtime)
    shutil.copy(final, out_path)
    return out_path


def _cli():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--character-image", required=True)
    parser.add_argument("--audio", required=True)
    parser.add_argument("--out", default=str(config.TMP_DIR / "talking_head.mp4"))
    parser.add_argument("--no-enhance", action="store_true")
    args = parser.parse_args()

    out = animate(args.character_image, args.audio, args.out, enhance=not args.no_enhance)
    print(f"wrote {out}")


if __name__ == "__main__":
    _cli()
