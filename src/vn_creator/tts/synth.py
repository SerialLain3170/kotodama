"""Japanese TTS via Style-Bert-VITS2 (JP-Extra), with emotion-style control.

The jvnv-F1-jp checkpoint was trained on the JVNV emotional-speech corpus,
whose style vectors are labelled by emotion (Neutral, Happy, Sad, Angry,
Surprised, Fear ...). We pick the closest style label to the scene's emotion
tag with a small keyword map, and fall back to Neutral.
"""
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from vn_creator import config

MODEL_DIR = config.MODELS_DIR / "tts" / "jvnv" / "jvnv-F1-jp"

# Keyword -> JVNV style name. Checked against style_vectors.npy's style2id at load time.
EMOTION_TO_STYLE_KEYWORDS = {
    "Happy": ["嬉し", "楽し", "笑", "喜"],
    "Sad": ["寂し", "悲し", "切な", "涙"],
    "Angry": ["怒", "苛立", "腹立"],
    "Surprise": ["驚", "びっくり"],
    "Fear": ["怖", "不安", "緊張"],
    "Disgust": ["嫌", "うんざり", "呆れ"],
}

_model = None


def _load_model():
    global _model
    if _model is not None:
        return _model

    from style_bert_vits2.tts_model import TTSModel
    from style_bert_vits2.nlp import bert_models
    from style_bert_vits2.constants import Languages

    jp_bert_repo = "ku-nlp/deberta-v2-large-japanese-char-wwm"
    # transformers>=5 respects a checkpoint's recorded torch_dtype (fp16 here) by
    # default, but the TTS net_g is fp32, so force the BERT model back to fp32
    # in place to avoid a half/float dtype mismatch during inference.
    bert_models.load_model(Languages.JP, jp_bert_repo).float()
    bert_models.load_tokenizer(Languages.JP, jp_bert_repo)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    _model = TTSModel(
        model_path=MODEL_DIR / "jvnv-F1-jp_e160_s14000.safetensors",
        config_path=MODEL_DIR / "config.json",
        style_vec_path=MODEL_DIR / "style_vectors.npy",
        device=device,
    )
    _model.load()
    return _model


def _pick_style(emotion_text: str, available_styles: list[str]) -> str:
    for style, keywords in EMOTION_TO_STYLE_KEYWORDS.items():
        if style not in available_styles:
            continue
        if any(kw in emotion_text for kw in keywords):
            return style
    return "Neutral" if "Neutral" in available_styles else available_styles[0]


def synthesize(
    text: str,
    emotion: str,
    target_duration_sec: float | None = None,
    out_path: str | Path = None,
) -> tuple[Path, float]:
    """Synthesize `text` as spoken Japanese. Returns (wav_path, actual_duration_sec).

    If target_duration_sec is given, adjusts the length scale (speaking rate)
    so the rendered clip roughly matches it, then re-synthesizes once.
    """
    model = _load_model()
    styles = list(model.hyper_parameters.data.style2id.keys())
    style = _pick_style(emotion, styles)

    length_scale = 1.0
    sr, audio = model.infer(text=text, style=style, length=length_scale)
    duration = len(audio) / sr

    if target_duration_sec and duration > 0:
        length_scale = max(0.6, min(1.8, target_duration_sec / duration))
        sr, audio = model.infer(text=text, style=style, length=length_scale)
        duration = len(audio) / sr

    out_path = Path(out_path) if out_path else config.TMP_DIR / "voice.wav"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(out_path, audio, sr)
    return out_path, duration


def _cli():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--text", required=True)
    parser.add_argument("--emotion", default="Neutral")
    parser.add_argument("--duration", type=float, default=None)
    parser.add_argument("--out", default=str(config.TMP_DIR / "voice.wav"))
    args = parser.parse_args()

    path, dur = synthesize(args.text, args.emotion, args.duration, args.out)
    print(f"wrote {path} ({dur:.2f}s)")


if __name__ == "__main__":
    _cli()
