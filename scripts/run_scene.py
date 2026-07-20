"""End-to-end orchestrator: scene inputs -> final composited mp4.

Default (auto-generation) path: Director -> TTS -> pad -> BGM -> one unified
1280x720 character+background illustration -> Ken Burns render. No animation
for now — this is a deliberate first step; force_shot_type still shapes the
Director's dialogue/mood (climactic vs normal beat) even though rendering is
currently the same either way.

If the caller supplies character_image and/or background_image explicitly,
the older separate-layer path is used instead (generate whichever is missing,
animate the character with SadTalker, composite over the background) — that
still exists for pre-made assets.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vn_creator import config
from vn_creator.animate.talking_head import animate
from vn_creator.bgm.generate import generate_bgm
from vn_creator.character.generate import generate_background, generate_character
from vn_creator.director.generate import generate_scene
from vn_creator.director.schema import SceneInput, SceneScript
from vn_creator.illustration.generate import generate_illustration
from vn_creator.render.compose import compose_scene
from vn_creator.render.illustration_shot import render_illustration_shot
from vn_creator.tts.synth import synthesize


def pad_audio(voice_path: Path, lead_sec: float, total_sec: float, out_path: Path) -> None:
    subprocess.run(
        [
            "ffmpeg", "-y", "-v", "error",
            "-i", str(voice_path),
            "-af", f"adelay={int(lead_sec * 1000)}:all=1,apad,atrim=0:{total_sec}",
            str(out_path),
        ],
        check=True,
    )


def _static_video_fallback(image_path: Path, audio_path: Path, duration_sec: float, out_path: Path) -> Path:
    """SadTalker's face detector (a real-face model) occasionally fails outright
    on some anime compositions even after lowering its confidence threshold —
    it's a genuine per-image gap, not something a fixed threshold can rule out.
    Rather than hard-failing the whole job, fall back to a static (non-animated)
    shot: the still image held for the full duration with the padded voice
    track muxed in, so compose_scene() can consume it exactly like a normal
    SadTalker output."""
    subprocess.run(
        [
            "ffmpeg", "-y", "-v", "error",
            "-loop", "1", "-i", str(image_path),
            "-i", str(audio_path),
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            "-t", str(duration_sec),
            "-shortest",
            str(out_path),
        ],
        check=True,
    )
    return out_path


def run(
    character_name: str,
    character_description: str,
    character_image: str | None,
    background_image: str | None,
    background_description: str,
    persona: str,
    context: str,
    genre: str,
    out_path: str,
    character_seed: int | None = None,
    force_shot_type: str | None = None,
    retro_style: bool = False,
    text_style: str | None = None,
) -> Path:
    # retro illustrations default to the Leaf-style full-screen NVL textbox
    # unless the caller picked a style explicitly.
    text_style = text_style or ("leaf_nvl" if retro_style else "default")
    scene_input = SceneInput(
        character_name=character_name,
        character_description=character_description,
        background_description=background_description,
        persona=persona,
        context=context,
        genre=genre,
    )

    print("[1/5] Scene Director: generating script...", file=sys.stderr)
    scene = generate_scene(scene_input, force_shot_type=force_shot_type)
    print(json.dumps(scene.model_dump(), ensure_ascii=False, indent=2), file=sys.stderr)

    print("[2/5] TTS: synthesizing voice...", file=sys.stderr)
    target_speech_dur = scene.speech_end - scene.speech_start
    raw_voice_path = config.TMP_DIR / "run_voice_raw.wav"
    _, actual_dur = synthesize(scene.dialogue, scene.emotion, target_speech_dur, raw_voice_path)

    # Reconcile the script's timeline with the TTS engine's actual output length.
    scene.speech_end = scene.speech_start + actual_dur
    scene.duration_sec = max(scene.duration_sec, scene.speech_end + 0.3)

    print("[3/5] Padding audio to full shot length...", file=sys.stderr)
    padded_audio_path = config.TMP_DIR / "run_voice_padded.wav"
    pad_audio(raw_voice_path, scene.speech_start, scene.duration_sec, padded_audio_path)

    print("[4/5] Generating BGM...", file=sys.stderr)
    bgm_path = generate_bgm(
        scene.bgm.mood, scene.bgm.instrument, scene.bgm.tempo_bpm, scene.duration_sec,
        config.TMP_DIR / "run_bgm.wav", retro=retro_style,
    )

    if character_image is not None or background_image is not None:
        # Caller supplied their own still(s) — keep the separate character/
        # background compositing path, since in that case the two images
        # really are independent assets by design.
        if character_image is None:
            print("[5/5] Character generation: generating anime still...", file=sys.stderr)
            character_image = generate_character(
                character_description, config.TMP_DIR / "run_character.png", seed=character_seed
            )
        if background_image is None:
            print("[5/5] Background generation: generating scene art...", file=sys.stderr)
            background_image = generate_background(
                background_description, config.TMP_DIR / "run_background.png"
            )
        print("[5/5] Animating and compositing final scene...", file=sys.stderr)
        try:
            performance_video = animate(character_image, padded_audio_path, config.TMP_DIR / "run_performance.mp4")
        except subprocess.CalledProcessError as e:
            print(
                f"WARNING: SadTalker failed ({e}); falling back to a static (non-animated) shot "
                "for this character image.", file=sys.stderr,
            )
            performance_video = _static_video_fallback(
                Path(character_image), padded_audio_path, scene.duration_sec,
                config.TMP_DIR / "run_performance.mp4",
            )
        final = compose_scene(
            scene, character_name, performance_video, background_image, bgm_path, out_path,
            text_style=text_style,
        )
    else:
        # Default auto-generation path: one unified character+background
        # illustration (1280x720, matches the final video exactly — no crop/
        # resize needed), rendered as a static Ken Burns shot. No animation for
        # now — SadTalker is disabled here as an initial simplification; the
        # static illustration alone is the first step.
        print("[5/5] Generating unified scene illustration...", file=sys.stderr)
        illustration_path = generate_illustration(
            scene.illustration_prompt, config.TMP_DIR / "run_illustration.png",
            seed=character_seed, retro=retro_style,
        )
        print("[5/5] Rendering Ken Burns shot...", file=sys.stderr)
        final = render_illustration_shot(
            scene, character_name, illustration_path, padded_audio_path, bgm_path, out_path,
            text_style=text_style,
            # True PC-98 640x400 + palette quantize/dither, and many PC-98-era
            # titles (esp. early Leaf works) had no voice acting at all.
            retro=retro_style, mute_voice=retro_style,
        )

    print(f"done: {final}", file=sys.stderr)
    return final


def _cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--character-name", required=True)
    parser.add_argument("--character-description", required=True)
    parser.add_argument("--character-image", default=None, help="Path to a still image; omit to generate one")
    parser.add_argument("--character-seed", type=int, default=None)
    parser.add_argument("--background-image", default=None, help="Path to a background image; omit to generate one")
    parser.add_argument("--background-description", required=True)
    parser.add_argument("--persona", required=True)
    parser.add_argument("--context", required=True)
    parser.add_argument("--genre", required=True)
    parser.add_argument("--force-shot-type", choices=["talking_head", "illustration"], default=None)
    parser.add_argument(
        "--retro-style", action="store_true",
        help="PC-98 look for illustration shots (style LoRA + palette/dither filter)",
    )
    parser.add_argument(
        "--text-style", default=None, choices=["default", "leaf_nvl", "ddlc"],
        help="Textbox style; defaults to leaf_nvl if --retro-style else default",
    )
    parser.add_argument("--out", default=str(config.OUTPUTS_DIR / "scene.mp4"))
    args = parser.parse_args()

    run(
        args.character_name, args.character_description, args.character_image,
        args.background_image, args.background_description,
        args.persona, args.context, args.genre, args.out,
        character_seed=args.character_seed,
        force_shot_type=args.force_shot_type,
        retro_style=args.retro_style,
        text_style=args.text_style,
    )


if __name__ == "__main__":
    _cli()
