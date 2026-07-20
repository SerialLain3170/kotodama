"""Unified character+background illustration generation for event-CG-style shots.

Uses Illustrious-XL (SDXL), which community consensus credits as the strongest
Danbooru-tag-trained checkpoint at inferring a contextual background from a
prompt rather than defaulting to a simple/portrait background — the specific
failure mode we hit with Animagine XL 4.0 for character portraits.
"""
from pathlib import Path

import torch

from vn_creator import config
from vn_creator.illustration.lora_merge import merge_text_encoder_lora

CHECKPOINT_PATH = config.MODELS_DIR / "illustration" / "Illustrious-XL-v2.0.safetensors"
VAE_PATH = config.MODELS_DIR / "illustration" / "sdxl-vae-fp16-fix"
LORA_DIR = config.MODELS_DIR / "illustration" / "loras"
PC98_LORA_PATH = LORA_DIR / "pc98_style_illustrious.safetensors"
PC98_TRIGGER_TAGS = "pc98, pixel art, retro game screenshot"
DEFAULT_LORA_STRENGTH = 1.6

# Countering the portrait/simple-background bias matters more than positive
# scenery tags (per community consensus on this checkpoint family).
NEGATIVE_PROMPT = (
    "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, "
    "fewer digits, cropped, worst quality, low quality, normal quality, "
    "jpeg artifacts, signature, watermark, username, blurry, multiple views, "
    "simple background, white background, plain background, close-up, "
    "portrait, face only"
)

_pipe = None
_merged_style_lora = None  # (path, strength) of the LoRA currently merged into _pipe, if any


def _load():
    global _pipe
    if _pipe is not None:
        return _pipe

    from diffusers import AutoencoderKL, StableDiffusionXLPipeline

    device = "cuda" if torch.cuda.is_available() else "cpu"
    vae = AutoencoderKL.from_pretrained(VAE_PATH, torch_dtype=torch.float16)
    _pipe = StableDiffusionXLPipeline.from_single_file(
        CHECKPOINT_PATH, vae=vae, torch_dtype=torch.float16, use_safetensors=True
    ).to(device)
    return _pipe


def _apply_style_lora(style_lora: str | Path, lora_strength: float) -> None:
    """Merge a style LoRA's text-encoder weights into the cached pipeline, once.
    Merging is in-place and not reversible, so this only supports one style LoRA
    (at one strength) per process — fine for the current one-shot-per-process CLI/
    orchestrator usage. Raises if a different LoRA/strength was already merged."""
    global _merged_style_lora
    style_lora = Path(style_lora)
    requested = (style_lora, lora_strength)
    if _merged_style_lora == requested:
        return
    if _merged_style_lora is not None:
        raise RuntimeError(
            f"A different style LoRA ({_merged_style_lora}) is already merged into "
            "this process's pipeline; merging is not reversible. Use a fresh process."
        )
    pipe = _load()
    applied = merge_text_encoder_lora(pipe, style_lora, multiplier=lora_strength)
    if applied == 0:
        raise RuntimeError(f"Style LoRA {style_lora} had no matching text-encoder layers.")
    _merged_style_lora = requested


SIZE_PRESETS = {
    # Standard 1280x720 canvas — matches the final video resolution exactly,
    # so no crop/resize is needed downstream.
    "wide": (1280, 720),
    # talking_head shots: the face gets cropped and animated by SadTalker, whose
    # underlying face detector (a real-face model, unreliable enough on anime
    # already) needs the face reasonably large and centered — a wide landscape
    # frame leaves the face small and off to one side, which is what caused
    # detection failures in practice. Portrait framing matches the aspect the
    # character-portrait pipeline already validated as SadTalker-friendly.
    "portrait": (896, 1152),
}


def generate_illustration(
    illustration_prompt: str,
    out_path: str | Path = None,
    seed: int | None = None,
    steps: int = 30,
    guidance_scale: float = 6.5,
    style_lora: str | Path | None = None,
    lora_strength: float | None = None,
    retro: bool = False,
    framing: str = "wide",
) -> Path:
    """illustration_prompt is expected to already describe character+background
    together (Danbooru-tag style), as produced by the Scene Director.

    style_lora: optional path to a kohya-format style LoRA (e.g. the PC-98 style
    LoRA in LORA_DIR) merged into the text encoders before generation.

    retro: shorthand for the PC-98 art style — merges PC98_LORA_PATH (unless
    style_lora is explicitly set to something else) and adds its trigger tags.
    Generation still happens at this module's normal resolution; the actual
    hardware-resolution/palette pixelation (640x400, limited colors) is applied
    later in render/illustration_shot.py, after the Ken Burns crop, so it
    matches the final output pixel grid exactly instead of being baked in here
    and then resampled.

    framing: "wide" (default, for illustration/event-CG shots) or "portrait"
    (for talking_head shots — see SIZE_PRESETS).
    """
    if retro:
        style_lora = style_lora or PC98_LORA_PATH
        illustration_prompt = f"{PC98_TRIGGER_TAGS}, {illustration_prompt}"
    lora_strength = lora_strength if lora_strength is not None else DEFAULT_LORA_STRENGTH

    if style_lora:
        _apply_style_lora(style_lora, lora_strength)
    pipe = _load()

    # The usual "modern ultra-detailed" quality boilerplate actively fights a
    # retro/low-fi style LoRA, so skip it when one is active.
    prefix = "" if style_lora else "masterpiece, best quality, very aesthetic, absurdres, "
    prompt = f"{prefix}{illustration_prompt}"

    generator = None
    if seed is not None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        generator = torch.Generator(device=device).manual_seed(seed)

    width, height = SIZE_PRESETS[framing]
    image = pipe(
        prompt=prompt,
        negative_prompt=NEGATIVE_PROMPT,
        width=width,
        height=height,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        generator=generator,
    ).images[0]

    out_path = Path(out_path) if out_path else config.TMP_DIR / "illustration.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path)
    return out_path


def _cli():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--out", default=str(config.TMP_DIR / "illustration.png"))
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--style-lora", default=None, help="Path to a kohya-format style LoRA")
    parser.add_argument("--lora-strength", type=float, default=None)
    parser.add_argument("--retro", action="store_true", help="PC-98 art style LoRA + trigger tags")
    parser.add_argument("--framing", default="wide", choices=["wide", "portrait"])
    args = parser.parse_args()

    out = generate_illustration(
        args.prompt, args.out, args.seed,
        style_lora=args.style_lora, lora_strength=args.lora_strength,
        retro=args.retro, framing=args.framing,
    )
    print(f"wrote {out}")


if __name__ == "__main__":
    _cli()
