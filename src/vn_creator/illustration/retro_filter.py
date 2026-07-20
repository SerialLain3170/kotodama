"""Algorithmic retro-hardware color filter: downscale -> palette quantize -> dither -> upscale.

Reproduces the actual technical constraint of limited-color-era hardware (e.g.
PC-98/VGA-era graphics) deterministically, rather than asking a diffusion model
to imitate it stochastically via prompting. Composable with any illustration
regardless of which checkpoint/LoRA generated it.
"""
from pathlib import Path

from PIL import Image


def quantize_dither(image: Image.Image, n_colors: int = 64, native_width: int | None = None) -> Image.Image:
    """In-memory core: downscale to native_width (no-op if already <= that width
    or native_width is None) -> palette quantize -> dither -> upscale back to
    the input size with nearest-neighbor. Called with native_width == image's
    own width (e.g. when the image IS already the target hardware resolution,
    such as a 640x400 frame), the resize steps are no-ops and this just quantizes
    at that native resolution — no faked blockiness, genuinely low-res output."""
    src = image.convert("RGB")
    if native_width and native_width < src.width:
        native_height = round(native_width * src.height / src.width)
        low_res = src.resize((native_width, native_height), Image.LANCZOS)
    else:
        low_res = src

    quantized = low_res.quantize(colors=n_colors, method=Image.MEDIANCUT, dither=Image.FLOYDSTEINBERG)
    return quantized.convert("RGB").resize(src.size, Image.NEAREST)


def apply_retro_filter(
    image_path: str | Path,
    out_path: str | Path = None,
    n_colors: int = 64,
    native_width: int = 320,
) -> Path:
    """File-based wrapper around quantize_dither. n_colors: 16 ~ EGA-era feel,
    64-256 ~ higher-end VGA/PC-98 EGC feel."""
    src = Image.open(image_path)
    result = quantize_dither(src, n_colors, native_width)

    out_path = Path(out_path) if out_path else Path(image_path).with_stem(Path(image_path).stem + "_retro")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.save(out_path)
    return out_path


def _cli():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--out", default=None)
    parser.add_argument("--colors", type=int, default=64)
    parser.add_argument("--native-width", type=int, default=320)
    args = parser.parse_args()

    out = apply_retro_filter(args.image, args.out, args.colors, args.native_width)
    print(f"wrote {out}")


if __name__ == "__main__":
    _cli()
