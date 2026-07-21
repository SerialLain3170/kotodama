"""Render a static event-CG illustration into a shot: Ken Burns pan/zoom + textbox + audio mix.

Unlike compose.py (which layers a separate background + an already-animated
character video), this takes ONE unified illustration and everything else is
computed here: the pan/zoom crop per frame, and the textbox overlay, are both
drawn directly with PIL onto the same frame so there's no compositing mismatch.
"""
import subprocess
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw

from vn_creator import config
from vn_creator.director.schema import CameraMove, SceneScript
from vn_creator.illustration.retro_filter import quantize_dither
from vn_creator.render.textbox import draw_textbox, make_text_context

RETRO_CANVAS_SIZE = (640, 400)  # true PC-9801 EGC 256-color mode resolution
FPS = 30
BGM_VOLUME = 0.6
VOICE_VOLUME = 0.75


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def _kenburns_crop(img: Image.Image, canvas_size: tuple[int, int], zoom: float, focus_x: float, focus_y: float) -> Image.Image:
    """Crop a `1/zoom`-sized window centered at the normalized focus point, then resize to canvas."""
    cw, ch = canvas_size
    iw, ih = img.size
    canvas_aspect = cw / ch

    # Largest crop window with the canvas aspect ratio that still fits in the image at this zoom.
    crop_h = ih / zoom
    crop_w = crop_h * canvas_aspect
    if crop_w > iw:
        crop_w = iw
        crop_h = crop_w / canvas_aspect

    cx, cy = focus_x * iw, focus_y * ih
    x0 = min(max(cx - crop_w / 2, 0), iw - crop_w)
    y0 = min(max(cy - crop_h / 2, 0), ih - crop_h)

    crop = img.crop((x0, y0, x0 + crop_w, y0 + crop_h))
    return crop.resize(canvas_size, Image.LANCZOS)


def render_illustration_shot(
    scene: SceneScript,
    character_name: str,
    illustration_path: str | Path,
    voice_audio_path: str | Path,
    bgm_path: str | Path,
    out_path: str | Path = None,
    text_style: str = "default",
    retro: bool = False,
    retro_n_colors: int = 64,
    mute_voice: bool = False,
) -> Path:
    """retro: render at the true PC-9801 EGC resolution (640x400, see
    RETRO_CANVAS_SIZE) instead of the illustration's native size, and
    palette-quantize/dither each frame AFTER the Ken Burns crop so art and text
    share one native pixel grid at the final output resolution — no
    upscale-from-a-smaller-fake-resolution trick needed, since 640x400 IS the
    resolution being reproduced. Generation (illustration/generate.py) still
    happens at that module's own resolution; this is the "resize to the size
    after generation" step. Non-retro shots keep the illustration's own
    resolution/aspect ratio (see canvas_size below) instead of forcing a fixed
    16:9 canvas.

    mute_voice: exclude the voice track from the final mix (BGM only) — many
    early PC-98 titles had no voice acting at all; scene.speech_start/end are
    still used for text-reveal timing regardless."""
    out_path = Path(out_path) if out_path else config.OUTPUTS_DIR / "scene.mp4"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    illustration = Image.open(illustration_path).convert("RGB")
    # Non-retro: keep the illustration's own native resolution/aspect ratio
    # instead of force-cropping into a fixed 16:9 canvas (which was zooming in
    # hard on portrait-framed talking_head illustrations). Retro is the one
    # case that SHOULD force a fixed resolution, since 640x400 is the actual
    # hardware mode being reproduced, not an arbitrary display size.
    canvas_size = RETRO_CANVAS_SIZE if retro else illustration.size
    # camera_move is only ever filled in by the Director for shot_type
    # "illustration" (see schema.py) — default to a static, no-op move (its
    # own field defaults already are 1.0 zoom / centered focus) so a
    # "talking_head"-labeled scene still renders fine here.
    cam = scene.camera_move or CameraMove()
    text_ctx = make_text_context(text_style, scene, canvas_size)

    with tempfile.TemporaryDirectory(dir=config.TMP_DIR) as tmp:
        tmp = Path(tmp)
        frames_dir = tmp / "frames"
        frames_dir.mkdir()

        n_frames = int(round(scene.duration_sec * FPS))
        for i in range(n_frames):
            t = i / FPS
            progress = min(1.0, t / scene.duration_sec) if scene.duration_sec > 0 else 0.0

            zoom = _lerp(cam.start_zoom, cam.end_zoom, progress)
            focus_x = _lerp(cam.start_focus_x, cam.end_focus_x, progress)
            focus_y = _lerp(cam.start_focus_y, cam.end_focus_y, progress)

            frame_rgb = _kenburns_crop(illustration, canvas_size, zoom, focus_x, focus_y)
            if retro:
                frame_rgb = quantize_dither(frame_rgb, n_colors=retro_n_colors, native_width=canvas_size[0])
            frame = frame_rgb.convert("RGBA")

            text_layer = Image.new("RGBA", canvas_size, (0, 0, 0, 0))
            draw_textbox(text_style, ImageDraw.Draw(text_layer), canvas_size, scene, character_name, t, text_ctx)
            composited = Image.alpha_composite(frame, text_layer)
            composited.convert("RGB").save(frames_dir / f"{i:05d}.png")

        mixed_audio = tmp / "mixed.wav"
        if mute_voice:
            subprocess.run(
                [
                    "ffmpeg", "-y", "-v", "error",
                    "-stream_loop", "-1", "-i", str(bgm_path),
                    "-af", f"atrim=0:{scene.duration_sec},volume={BGM_VOLUME}",
                    "-t", str(scene.duration_sec),
                    str(mixed_audio),
                ],
                check=True,
            )
        else:
            subprocess.run(
                [
                    "ffmpeg", "-y", "-v", "error",
                    "-i", str(voice_audio_path),
                    "-stream_loop", "-1", "-i", str(bgm_path),
                    "-filter_complex",
                    f"[1:a]atrim=0:{scene.duration_sec},volume={BGM_VOLUME}[bgm];"
                    f"[0:a]atrim=0:{scene.duration_sec},volume={VOICE_VOLUME}[voice];"
                    "[bgm][voice]amix=inputs=2:duration=first:dropout_transition=0[aout]",
                    "-map", "[aout]",
                    "-t", str(scene.duration_sec),
                    str(mixed_audio),
                ],
                check=True,
            )

        subprocess.run(
            [
                "ffmpeg", "-y", "-v", "error",
                "-framerate", str(FPS), "-i", str(frames_dir / "%05d.png"),
                "-i", str(mixed_audio),
                "-map", "0:v", "-map", "1:a",
                "-t", str(scene.duration_sec),
                "-r", str(FPS),
                "-c:v", "libx264", "-pix_fmt", "yuv420p",
                "-c:a", "aac",
                str(out_path),
            ],
            check=True,
        )

    return out_path


def _cli():
    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-json", required=True)
    parser.add_argument("--character-name", required=True)
    parser.add_argument("--illustration", required=True)
    parser.add_argument("--voice", required=True)
    parser.add_argument("--bgm", required=True)
    parser.add_argument("--out", default=str(config.OUTPUTS_DIR / "scene.mp4"))
    parser.add_argument("--text-style", default="default", choices=["default", "leaf_nvl", "ddlc"])
    parser.add_argument("--retro", action="store_true", help="Render at true PC-98 640x400 with palette quantize/dither")
    parser.add_argument("--retro-colors", type=int, default=64)
    parser.add_argument("--mute-voice", action="store_true")
    args = parser.parse_args()

    scene = SceneScript.model_validate(json.loads(Path(args.scene_json).read_text()))
    out = render_illustration_shot(
        scene, args.character_name, args.illustration, args.voice, args.bgm, args.out,
        text_style=args.text_style, retro=args.retro, retro_n_colors=args.retro_colors,
        mute_voice=args.mute_voice,
    )
    print(f"wrote {out}")


if __name__ == "__main__":
    _cli()
