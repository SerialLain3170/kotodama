"""BGM generation via MusicGen, conditioned on a short mood/instrument/tempo prompt."""
from pathlib import Path

import scipy.io.wavfile
import torch

from vn_creator import config

MODEL_ID = "facebook/musicgen-medium"
PC98_BGM_TAGS = "YM2608 FM synthesis, PC-98 chiptune, square wave PSG arpeggio, OPNA, retro Japanese computer game music"

_model = None
_processor = None


def _load():
    global _model, _processor
    if _model is not None:
        return _model, _processor

    from transformers import AutoProcessor, MusicgenForConditionalGeneration

    device = "cuda" if torch.cuda.is_available() else "cpu"
    _processor = AutoProcessor.from_pretrained(MODEL_ID)
    _model = MusicgenForConditionalGeneration.from_pretrained(MODEL_ID).to(device)
    return _model, _processor


def generate_bgm(
    mood: str,
    instrument: str,
    tempo_bpm: int,
    duration_sec: float,
    out_path: str | Path = None,
    retro: bool = False,
) -> Path:
    """retro: bias the generation toward PC-98/YM2608-era FM-synth chiptune
    sound by adding those descriptor tags to the prompt, layered on top of the
    scene's own mood/instrument/tempo rather than replacing them."""
    model, processor = _load()
    device = next(model.parameters()).device

    tags = f"{instrument}, {PC98_BGM_TAGS}" if retro else instrument
    prompt = f"{mood}, {tags}, {tempo_bpm} bpm, instrumental, no vocals, loopable"
    inputs = processor(text=[prompt], padding=True, return_tensors="pt").to(device)

    frame_rate = model.config.audio_encoder.frame_rate
    max_new_tokens = int(duration_sec * frame_rate)

    with torch.no_grad():
        audio = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=True, guidance_scale=3.0)

    sr = model.config.audio_encoder.sampling_rate
    audio = audio[0, 0].cpu().numpy()

    out_path = Path(out_path) if out_path else config.TMP_DIR / "bgm.wav"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    scipy.io.wavfile.write(out_path, rate=sr, data=audio)
    return out_path


def _cli():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--mood", required=True)
    parser.add_argument("--instrument", required=True)
    parser.add_argument("--tempo", type=int, required=True)
    parser.add_argument("--duration", type=float, required=True)
    parser.add_argument("--out", default=str(config.TMP_DIR / "bgm.wav"))
    parser.add_argument("--retro", action="store_true", help="Bias toward PC-98/YM2608 FM-synth chiptune sound")
    args = parser.parse_args()

    out = generate_bgm(args.mood, args.instrument, args.tempo, args.duration, args.out, retro=args.retro)
    print(f"wrote {out}")


if __name__ == "__main__":
    _cli()
