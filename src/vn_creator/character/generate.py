"""Anime character still generation via Animagine XL 4.0 (SDXL).

Produces a front-facing bust-up portrait suitable as a SadTalker driving
image: centered face, simple background, neutral-ish default expression
unless the persona/emotion calls for something else.
"""
from pathlib import Path

import torch

from vn_creator import config

MODEL_DIR = config.MODELS_DIR / "character" / "animagine-xl-4.0"

NEGATIVE_PROMPT = (
    "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, "
    "fewer digits, cropped, worst quality, low quality, normal quality, "
    "jpeg artifacts, signature, watermark, username, blurry, multiple views, "
    "full body, wide shot, extreme close-up, close-up, face only, out of frame, "
    "head cut off, cropped head"
)

_pipe = None


def _load():
    global _pipe
    if _pipe is not None:
        return _pipe

    from diffusers import StableDiffusionXLPipeline

    device = "cuda" if torch.cuda.is_available() else "cpu"
    _pipe = StableDiffusionXLPipeline.from_pretrained(
        MODEL_DIR, torch_dtype=torch.float16, use_safetensors=True
    ).to(device)
    return _pipe


def generate_character(
    character_description: str,
    out_path: str | Path = None,
    seed: int | None = None,
    steps: int = 28,
    guidance_scale: float = 6.5,
) -> Path:
    pipe = _load()

    prompt = (
        "masterpiece, best quality, very aesthetic, absurdres, "
        f"1girl, solo, {character_description}, upper body, medium shot, cowboy shot, "
        "looking at viewer, front view, simple background, headroom, "
        "detailed face, anime screencap"
    )

    generator = None
    if seed is not None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        generator = torch.Generator(device=device).manual_seed(seed)

    image = pipe(
        prompt=prompt,
        negative_prompt=NEGATIVE_PROMPT,
        width=832,
        height=1216,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        generator=generator,
    ).images[0]

    out_path = Path(out_path) if out_path else config.TMP_DIR / "character.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path)
    return out_path


def generate_background(
    background_description: str,
    out_path: str | Path = None,
    seed: int | None = None,
    steps: int = 28,
    guidance_scale: float = 6.5,
) -> Path:
    pipe = _load()

    prompt = (
        "masterpiece, best quality, very aesthetic, absurdres, "
        f"scenery, background, no humans, {background_description}, "
        "anime background art, wide shot, detailed"
    )
    negative_prompt = (
        "lowres, worst quality, low quality, normal quality, jpeg artifacts, "
        "signature, watermark, username, blurry, 1girl, 1boy, person, human, "
        "text, extra digit"
    )

    generator = None
    if seed is not None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        generator = torch.Generator(device=device).manual_seed(seed)

    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=1344,
        height=768,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        generator=generator,
    ).images[0]

    out_path = Path(out_path) if out_path else config.TMP_DIR / "background.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path)
    return out_path


def _cli():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--description", required=True)
    parser.add_argument("--out", default=str(config.TMP_DIR / "character.png"))
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    out = generate_character(args.description, args.out, args.seed)
    print(f"wrote {out}")


if __name__ == "__main__":
    _cli()
