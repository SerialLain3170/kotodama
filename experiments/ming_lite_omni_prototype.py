"""Prototype #2: a more realistic near-term substitute for the Director +
illustration + BGM stages, using TWO small models instead of one giant one.

Why not just Ming-Flash-Omni-2.0 (prototype #1, ming_omni_prototype.py): that
model is 100B/6B-active MoE (~200GB bf16) and doesn't fit on this machine
(4x L40S = 184GB total, and it's shared with other tenants). This prototype
swaps in:

  - Ming-Lite-Omni (inclusionAI/Ming-Lite-Omni-1.5, 20.3B total / 3B active) --
    one model, text + image generation. ~42GB bf16, fits on a single L40S.
    https://huggingface.co/inclusionAI/Ming-Lite-Omni-1.5
  - Ming-omni-tts-0.5B (inclusionAI/Ming-omni-tts-0.5B) -- tiny (~1GB bf16),
    and unlike Ming-Flash-Omni's talker, this one has a genuine standalone
    instrumental-music mode (prompt="Please generate music based on the
    following description.", no speech text) -- a real match for this
    project's existing BGM stage (bgm/generate.py), not speech-with-BGM-
    underneath. https://github.com/inclusionAI/Ming-omni-tts

This is still two models, not one -- so it doesn't test the "one omni model
for everything" hypothesis as directly as prototype #1 would. What it DOES
test cheaply: whether Ming-Lite-Omni's *joint* story+image generation (one
model, one forward context) produces better-matched art-to-narration
coherence than this project's current Director-LLM-writes-a-prompt-then-a-
separate-SDXL-renders-it split. That's the more affordable half of the
original hypothesis to check first.

STATUS: UNRUN. Ming-Lite-Omni's own README states ~42G GPU memory in
bfloat16 tested on H800-80GB/H20-96G -- tight on a 46GB L40S with nothing
else resident, and all 4 GPUs on this box were nearly full (other tenants)
when this was written. Ming-omni-tts-0.5B is small enough to fit anywhere.
Use --dry-run to sanity-check the scene/prompts without loading anything.

Known unknowns:
  - The image-generation call (model.generate(**inputs, image_gen=True,
    image_gen_seed=...)) is VERIFIED for Ming-Flash-Omni-2.0
    (test_infer_imagegen.py) but only ASSUMED to carry over unchanged to
    Ming-Lite-Omni-1.5 -- same author lineage (BailingMM -> BailingMM2), but
    not independently confirmed for this specific checkpoint. First thing to
    check on a real run.
  - Setup requires `git clone https://github.com/inclusionAI/Ming` (for
    modeling_bailingmm.py) same as prototype #1's Ming-Flash-Omni clone, plus
    a SEPARATE `git clone https://github.com/inclusionAI/Ming-omni-tts` for
    the BGM model's modeling_bailingmm.py/AudioVAE code -- these are two
    distinct repos with similarly-named files; do not mix them up on
    sys.path.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vn_creator import config  # noqa: E402
from sample_scene import SAMPLE_SCENE, ILLUSTRATION_PROMPT, STORY_PROMPT_JA  # noqa: E402

LITE_OMNI_CODE_DIR = config.MODELS_DIR / "ming_omni" / "Ming"  # same clone prototype #1 uses
LITE_OMNI_MODEL_ID = "inclusionAI/Ming-Lite-Omni-1.5"
OMNI_TTS_CODE_DIR = config.MODELS_DIR / "ming_omni" / "Ming-omni-tts"  # separate repo
OMNI_TTS_MODEL_ID = "inclusionAI/Ming-omni-tts-0.5B"
OUT_DIR = config.TMP_DIR / "ming_lite_omni_prototype"

_lite_model = None
_lite_processor = None
_tts_model = None


def _load_lite_omni(load_image_gen: bool):
    global _lite_model, _lite_processor
    if _lite_model is not None:
        return _lite_model, _lite_processor

    if str(LITE_OMNI_CODE_DIR) not in sys.path:
        if not LITE_OMNI_CODE_DIR.exists():
            raise RuntimeError(
                f"{LITE_OMNI_CODE_DIR} not found. Clone https://github.com/inclusionAI/Ming there first."
            )
        sys.path.insert(0, str(LITE_OMNI_CODE_DIR))

    import torch
    from transformers import AutoProcessor
    from modeling_bailingmm import BailingMMNativeForConditionalGeneration

    _lite_model = BailingMMNativeForConditionalGeneration.from_pretrained(
        LITE_OMNI_MODEL_ID,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        load_image_gen=load_image_gen,
        low_cpu_mem_usage=True,
    ).to("cuda")
    _lite_processor = AutoProcessor.from_pretrained(LITE_OMNI_CODE_DIR, trust_remote_code=True)
    return _lite_model, _lite_processor


def _load_omni_tts():
    global _tts_model
    if _tts_model is not None:
        return _tts_model

    if str(OMNI_TTS_CODE_DIR) not in sys.path:
        if not OMNI_TTS_CODE_DIR.exists():
            raise RuntimeError(
                f"{OMNI_TTS_CODE_DIR} not found. Clone https://github.com/inclusionAI/Ming-omni-tts there first "
                "(a separate repo from Ming-Flash-Omni/Ming-Lite-Omni's -- don't reuse that clone)."
            )
        sys.path.insert(0, str(OMNI_TTS_CODE_DIR))

    # MingAudio wraps model loading + generation; copied from
    # Ming-omni-tts/cookbooks/test.py rather than reimplemented, since its
    # constructor also loads the speaker-embedding extractor and text
    # normalizer config relative to that repo's own directory.
    from cookbooks.test import MingAudio

    _tts_model = MingAudio(OMNI_TTS_MODEL_ID)
    return _tts_model


def generate_story(max_new_tokens: int = 512) -> str:
    """Same prompt as prototype #1, so outputs are comparable across models."""
    import torch

    model, processor = _load_lite_omni(load_image_gen=False)
    messages = [{"role": "HUMAN", "content": [{"type": "text", "text": STORY_PROMPT_JA}]}]

    text = processor.apply_chat_template(messages, add_generation_prompt=True)
    image_inputs, video_inputs, audio_inputs = processor.process_vision_info(messages)
    inputs = processor(
        text=[text], images=image_inputs, videos=video_inputs, audios=audio_inputs,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs, max_new_tokens=max_new_tokens, use_cache=True,
            eos_token_id=processor.gen_terminator,
        )
    trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)]
    return processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]


def generate_illustration(illustration_prompt: str, out_path: Path, seed: int = 42) -> Path:
    """Unverified for this checkpoint -- see module docstring's known unknowns."""
    model, processor = _load_lite_omni(load_image_gen=True)
    messages = [{"role": "HUMAN", "content": [{"type": "text", "text": illustration_prompt}]}]

    text = processor.apply_chat_template(messages, add_generation_prompt=True)
    image_inputs, _, _ = processor.process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, return_tensors="pt").to(model.device)

    image = model.generate(**inputs, image_gen=True, image_gen_seed=seed)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path)
    return out_path


def generate_bgm(mood: str, instrument: str, duration_sec: int, out_path: Path) -> Path:
    """Standalone instrumental BGM -- no speech text, no speaker -- via
    Ming-omni-tts's dedicated music mode. Verified against
    Ming-omni-tts/cookbooks/test.py's own "# BGM" example (attr dict ->
    a plain description string -> speech_generation() with a music prompt)."""
    tts = _load_omni_tts()
    attrs = {"Mood": mood, "Instrument": instrument, "Duration": f"{duration_sec}s."}
    text = " " + " ".join(f"{k}: {v}" for k, v in attrs.items())

    tts.speech_generation(
        prompt="Please generate music based on the following description.\n",
        text=text,
        max_decode_steps=400,
        output_wav_path=str(out_path),
    )
    return out_path


def run(dry_run: bool = False) -> None:
    scene = SAMPLE_SCENE
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if dry_run:
        print("[dry-run] would generate story for scene:", json.dumps(scene, ensure_ascii=False, indent=2))
        print("[dry-run] would generate illustration for prompt:", ILLUSTRATION_PROMPT)
        print("[dry-run] would generate standalone BGM with mood:", scene["bgm_mood"], scene["bgm_instrument"])
        print(f"[dry-run] outputs would be written under {OUT_DIR}")
        return

    t0 = time.time()
    print("[1/3] story (Ming-Lite-Omni)...", file=sys.stderr)
    story = generate_story()
    (OUT_DIR / "story.txt").write_text(story, encoding="utf-8")
    print(story)
    print(f"  ({time.time() - t0:.1f}s)", file=sys.stderr)

    t0 = time.time()
    print("[2/3] illustration (Ming-Lite-Omni)...", file=sys.stderr)
    illustration_path = generate_illustration(ILLUSTRATION_PROMPT, OUT_DIR / "illustration.jpg")
    print(f"  wrote {illustration_path} ({time.time() - t0:.1f}s)", file=sys.stderr)

    t0 = time.time()
    print("[3/3] BGM (Ming-omni-tts-0.5B)...", file=sys.stderr)
    bgm_path = generate_bgm(scene["bgm_mood"], scene["bgm_instrument"], duration_sec=20, out_path=OUT_DIR / "bgm.wav")
    print(f"  wrote {bgm_path} ({time.time() - t0:.1f}s)", file=sys.stderr)


def _cli():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print the scene/prompts that would be sent without loading either model",
    )
    args = parser.parse_args()
    run(dry_run=args.dry_run)


if __name__ == "__main__":
    _cli()
