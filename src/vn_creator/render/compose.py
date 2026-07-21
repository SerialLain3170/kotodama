"""Compositing: background + character performance video + textbox + audio mix -> final mp4.

The character performance video is expected to already span the full shot
duration (silence-padded audio in, so the character idles before/after the
line) and to carry the voice track. This module lays it over a background,
draws a typewriter-timed textbox with PIL (for correct Japanese text
rendering), and mixes in ducked BGM.
"""
import shutil
import subprocess
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw

from vn_creator import config
from vn_creator.director.schema import SceneScript
from vn_creator.render.textbox import draw_textbox, make_text_context

CANVAS_SIZE = (1280, 720)
FPS = 30
BGM_VOLUME = 0.6
VOICE_VOLUME = 0.75


def _render_textbox_frames(
    scene: SceneScript, character_name: str, frames_dir: Path, text_style: str, canvas_size: tuple[int, int]
) -> None:
    frames_dir.mkdir(parents=True, exist_ok=True)
    ctx = make_text_context(text_style, scene, canvas_size)

    n_frames = int(round(scene.duration_sec * FPS))
    for i in range(n_frames):
        t = i / FPS
        img = Image.new("RGBA", canvas_size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        draw_textbox(text_style, draw, canvas_size, scene, character_name, t, ctx)
        img.save(frames_dir / f"{i:05d}.png")


def _ffprobe_size(video_path: str | Path) -> tuple[int, int]:
    out = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=width,height", "-of", "csv=p=0",
            str(video_path),
        ],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    w, h = out.split(",")
    return int(w), int(h)


def _mix_audio(performance_video: str | Path, bgm_path: str | Path, duration_sec: float, out_path: Path) -> None:
    subprocess.run(
        [
            "ffmpeg", "-y", "-v", "error",
            "-i", str(performance_video),
            "-stream_loop", "-1", "-i", str(bgm_path),
            "-filter_complex",
            f"[1:a]atrim=0:{duration_sec},volume={BGM_VOLUME}[bgm];"
            f"[0:a]atrim=0:{duration_sec},volume={VOICE_VOLUME}[voice];"
            "[bgm][voice]amix=inputs=2:duration=first:dropout_transition=0[aout]",
            "-map", "[aout]",
            "-t", str(duration_sec),
            str(out_path),
        ],
        check=True,
    )


def compose_scene(
    scene: SceneScript,
    character_name: str,
    performance_video: str | Path,
    background_image: str | Path,
    bgm_path: str | Path,
    out_path: str | Path = None,
    text_style: str = "default",
) -> Path:
    """performance_video is a character-only clip (e.g. a portrait-only SadTalker
    render) that gets layered on top of a SEPARATELY generated background_image.
    Use this only when the caller supplied its own pre-made character/background
    stills — for auto-generated scenes, compose_unified_shot avoids compositing
    two independently-generated layers together (which is what causes a visible
    seam between character and background)."""
    out_path = Path(out_path) if out_path else config.OUTPUTS_DIR / "scene.mp4"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(dir=config.TMP_DIR) as tmp:
        tmp = Path(tmp)
        frames_dir = tmp / "textbox_frames"
        _render_textbox_frames(scene, character_name, frames_dir, text_style, CANVAS_SIZE)

        w, h = CANVAS_SIZE
        char_h = h
        mixed_audio = tmp / "mixed.wav"
        _mix_audio(performance_video, bgm_path, scene.duration_sec, mixed_audio)

        # composite background + character video + textbox frames, mux with mixed audio
        subprocess.run(
            [
                "ffmpeg", "-y", "-v", "error",
                "-loop", "1", "-i", str(background_image),
                "-i", str(performance_video),
                "-framerate", str(FPS), "-i", str(frames_dir / "%05d.png"),
                "-i", str(mixed_audio),
                "-filter_complex",
                f"[0:v]scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h}[bg];"
                f"[1:v]scale=-1:{char_h}[char];"
                "[bg][char]overlay=(W-w)/2:0[scene];"
                "[scene][2:v]overlay=0:0[vout]",
                "-map", "[vout]", "-map", "3:a",
                "-t", str(scene.duration_sec),
                "-r", str(FPS),
                "-c:v", "libx264", "-pix_fmt", "yuv420p",
                "-c:a", "aac",
                str(out_path),
            ],
            check=True,
        )

    return out_path


def compose_unified_shot(
    scene: SceneScript,
    character_name: str,
    performance_video: str | Path,
    bgm_path: str | Path,
    out_path: str | Path = None,
    text_style: str = "default",
) -> Path:
    """performance_video already contains the whole scene (character AND
    background baked into one illustration by illustration/generate.py, then
    animated in place by SadTalker — which crops just the face, animates it,
    and pastes it back into the same frame via seamlessClone). There's no
    second layer to composite, so this just overlays the textbox and mixes
    audio — no forced resize/crop to a fixed canvas; the output keeps
    performance_video's own resolution and aspect ratio."""
    out_path = Path(out_path) if out_path else config.OUTPUTS_DIR / "scene.mp4"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    canvas_size = _ffprobe_size(performance_video)

    with tempfile.TemporaryDirectory(dir=config.TMP_DIR) as tmp:
        tmp = Path(tmp)
        frames_dir = tmp / "textbox_frames"
        _render_textbox_frames(scene, character_name, frames_dir, text_style, canvas_size)

        mixed_audio = tmp / "mixed.wav"
        _mix_audio(performance_video, bgm_path, scene.duration_sec, mixed_audio)

        subprocess.run(
            [
                "ffmpeg", "-y", "-v", "error",
                "-i", str(performance_video),
                "-framerate", str(FPS), "-i", str(frames_dir / "%05d.png"),
                "-i", str(mixed_audio),
                "-filter_complex", "[0:v][1:v]overlay=0:0[vout]",
                "-map", "[vout]", "-map", "2:a",
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
    parser.add_argument("--performance-video", required=True)
    parser.add_argument("--background", default=None, help="Omit with --unified")
    parser.add_argument("--bgm", required=True)
    parser.add_argument("--out", default=str(config.OUTPUTS_DIR / "scene.mp4"))
    parser.add_argument("--text-style", default="default", choices=["default", "leaf_nvl", "ddlc"])
    parser.add_argument("--unified", action="store_true", help="performance_video already has the full scene baked in")
    args = parser.parse_args()

    scene = SceneScript.model_validate(json.loads(Path(args.scene_json).read_text()))
    if args.unified:
        out = compose_unified_shot(
            scene, args.character_name, args.performance_video, args.bgm, args.out,
            text_style=args.text_style,
        )
    else:
        out = compose_scene(
            scene, args.character_name, args.performance_video, args.background, args.bgm, args.out,
            text_style=args.text_style,
        )
    print(f"wrote {out}")


if __name__ == "__main__":
    _cli()
