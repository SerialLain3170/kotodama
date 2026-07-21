"""Retro (PC-98) illustration generation via Anima v1.0 + the PC-98 Gal LoRA.

Anima (circlestone-labs, NVIDIA Cosmos-Predict2 backbone + Qwen3 text encoder)
is a separate backend from illustration/generate.py's Illustrious-XL (SDXL) —
used only for retro-mode shots. The PC-98 Gal LoRA
(civitai.com/models/2715416) was trained on 87 images from 17 actual PC-98
titles, giving genuinely period-accurate dithering/linework/color instead of
Illustrious-XL's modern look plus an algorithmic palette filter. Unlike the
SDXL LoRA case (illustration/lora_merge.py, text-encoder-only), this LoRA
targets the diffusion transformer itself and merges in full (280/280 modules
— see anima_lora_merge.py), so it carries its whole intended style effect.
"""
from pathlib import Path

import torch

from vn_creator import config
from vn_creator.illustration.anima_lora_merge import merge_cosmos_transformer_lora

BASE_MODEL_DIR = config.MODELS_DIR / "illustration_anima" / "base"
PC98_GAL_LORA_PATH = config.MODELS_DIR / "illustration_anima" / "loras" / "pc98gal_style-v02_anima.safetensors"
PC98_GAL_TRIGGER = "pc98gal_style"
DEFAULT_LORA_STRENGTH = 1.0

NEGATIVE_PROMPT = (
    "modern anime, 3d, photorealistic, blurry, low quality, worst quality, "
    "simple background, white background, plain background, text, watermark, "
    "signature, extra digits, bad anatomy"
)

_pipe = None
_merged_lora = None  # (path, strength) currently merged into _pipe.transformer, if any


def _load():
    global _pipe
    if _pipe is not None:
        return _pipe

    from diffusers import ModularPipeline

    _pipe = ModularPipeline.from_pretrained(str(BASE_MODEL_DIR))
    _pipe.load_components(torch_dtype=torch.bfloat16)
    _pipe.to("cuda" if torch.cuda.is_available() else "cpu")
    return _pipe


def _apply_lora(lora_path: str | Path, lora_strength: float) -> None:
    global _merged_lora
    lora_path = Path(lora_path)
    requested = (lora_path, lora_strength)
    if _merged_lora == requested:
        return
    if _merged_lora is not None:
        raise RuntimeError(
            f"A different LoRA ({_merged_lora}) is already merged into this process's "
            "transformer; merging is not reversible. Use a fresh process."
        )
    pipe = _load()
    applied = merge_cosmos_transformer_lora(pipe.transformer, lora_path, multiplier=lora_strength)
    if applied == 0:
        raise RuntimeError(f"LoRA {lora_path} had no matching transformer blocks.")
    _merged_lora = requested


def generate_illustration_anima(
    illustration_prompt: str,
    out_path: str | Path = None,
    seed: int | None = None,
    steps: int = 30,
    width: int = 1280,
    height: int = 720,
    lora_path: str | Path = PC98_GAL_LORA_PATH,
    lora_strength: float | None = None,
) -> Path:
    """illustration_prompt should already describe character+background
    together, as produced by the Scene Director — same convention as
    illustration/generate.py's generate_illustration()."""
    lora_strength = lora_strength if lora_strength is not None else DEFAULT_LORA_STRENGTH
    if lora_path:
        _apply_lora(lora_path, lora_strength)
    pipe = _load()

    prompt = f"{PC98_GAL_TRIGGER}, {illustration_prompt}" if lora_path == PC98_GAL_LORA_PATH else illustration_prompt

    generator = None
    if seed is not None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        generator = torch.Generator(device=device).manual_seed(seed)

    image = pipe(
        prompt=prompt,
        negative_prompt=NEGATIVE_PROMPT,
        width=width,
        height=height,
        num_inference_steps=steps,
        generator=generator,
    ).images[0]

    out_path = Path(out_path) if out_path else config.TMP_DIR / "illustration_anima.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path)
    return out_path


def _cli():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--out", default=str(config.TMP_DIR / "illustration_anima.png"))
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--lora-strength", type=float, default=None)
    args = parser.parse_args()

    out = generate_illustration_anima(
        args.prompt, args.out, args.seed, width=args.width, height=args.height,
        lora_strength=args.lora_strength,
    )
    print(f"wrote {out}")


if __name__ == "__main__":
    _cli()
