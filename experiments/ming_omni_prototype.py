"""Prototype: test Ming-Flash-Omni-2.0 as a single-model replacement for the
Director LLM + illustration + BGM stages of the main pipeline, on one sample
scene.

Why this model: of the any-to-any candidates surveyed (CoDi-2, NExT-GPT,
Unified-IO 2, Emu3, Show-o, M2UGen, ...), Ming-Flash-Omni-2.0
(inclusionAI/Ming-flash-omni-2.0) is the only one found that genuinely
generates image + audio/music + text from one checkpoint rather than routing
to separate expert models. See https://huggingface.co/inclusionAI/Ming-flash-omni-2.0
and https://github.com/inclusionAI/Ming.

STATUS: UNRUN. This is 100B total / 6B active params (MoE) -- bf16 weights
alone are ~200GB. This machine has 4x L40S = 184GB total VRAM, and even that
theoretical max is under the model's footprint before counting the image-gen
and talker heads or KV cache; the GPUs are also shared with other tenants and
were nearly full when this was written. Do not run this until either more
VRAM is available (rented cloud GPUs) or a quantized checkpoint exists.
Use --dry-run to sanity-check the scene/prompts without loading anything.

Setup (once VRAM is available):
    git clone https://github.com/inclusionAI/Ming <MODELS_DIR>/ming_omni/Ming
  The HF repo ships weights only; the custom model classes
  (BailingMM2NativeForConditionalGeneration etc.) come from that GitHub repo,
  per the model's own test scripts (test_infer.py, test_infer_imagegen.py) --
  this script imports them the same way rather than trusting
  AutoModel(..., trust_remote_code=True) to resolve them, since that path
  isn't demonstrated anywhere in the upstream repo.

Known unknowns to resolve on first real run (this combination of image_gen +
talker in one process is NOT demonstrated upstream -- their test scripts
exercise text-gen, image-gen, and TTS/BGM as three separate standalone
programs, each with its own device_map strategy):
  - Whether _split_model()'s device layout (borrowed from their image-gen
    test script) is still correct once the talker submodule is also loaded.
  - Whether load_image_gen=True and load_talker=True can coexist in one
    from_pretrained() call at all (untested combination).
  - The BGM path (talker.instruct_audio_generation with an
    instruction={"BGM": {...}} field) generates narrated SPEECH with
    background music under it -- there is no documented standalone
    instrumental-only track mode like MusicGen's. That's a real mismatch
    with this project's BGM stage (bgm/generate.py produces vocal-free,
    loopable instrumental beds); this prototype generates a narration
    line + BGM together to see how close the BGM component actually gets
    to a usable instrumental bed, not as a drop-in MusicGen replacement.
"""
import argparse
import json
import os
import sys
import time
from bisect import bisect_left
from pathlib import Path

# This model needs multiple GPUs; config.py's default CUDA_VISIBLE_DEVICES=1
# pin (right for the rest of the single-GPU pipeline) must be overridden
# *before* vn_creator.config is imported. Claims all 4 shared GPUs by
# default -- this WILL contend with other tenants' jobs.
os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("VN_MING_GPUS", "0,1,2,3")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vn_creator import config  # noqa: E402
from sample_scene import SAMPLE_SCENE, ILLUSTRATION_PROMPT, STORY_PROMPT_JA  # noqa: E402

MING_CODE_DIR = config.MODELS_DIR / "ming_omni" / "Ming"
MODEL_ID = "inclusionAI/Ming-flash-omni-2.0"
OUT_DIR = config.TMP_DIR / "ming_omni_prototype"

_model = None
_processor = None


def _split_model(num_gpus: int, num_layers: int = 32) -> dict:
    """Layer-sharding device_map, copied from the model's own
    test_infer_imagegen.py: GPU 0 holds vision/audio/proj/embeddings/lm_head,
    the LLM's transformer layers are spread across the remaining GPUs. Needs
    >=2 GPUs. Unverified whether this still applies once the talker is also
    loaded (see module docstring)."""
    if num_gpus < 2:
        raise RuntimeError("_split_model needs >=2 visible GPUs; got 1. Set VN_MING_GPUS.")
    world_size = num_gpus - 1
    layer_per_gpu = num_layers // world_size
    boundaries = [i * layer_per_gpu - 1 for i in range(1, world_size + 1)]
    device_map = {}
    for i in range(num_layers):
        device_id = bisect_left(boundaries, i) + 1
        if device_id > world_size:
            device_id = i % world_size + 1
        device_map[f"model.model.layers.{i}"] = device_id
    for key in (
        "vision", "audio", "linear_proj", "linear_proj_audio",
        "model.model.word_embeddings.weight", "model.model.norm.weight",
        "model.lm_head.weight", "model.model.norm",
    ):
        device_map[key] = 0
    device_map[f"model.model.layers.{num_layers - 1}"] = 0
    return device_map


def _load(load_image_gen: bool, load_talker: bool):
    global _model, _processor
    if _model is not None:
        return _model, _processor

    if str(MING_CODE_DIR) not in sys.path:
        if not MING_CODE_DIR.exists():
            raise RuntimeError(
                f"{MING_CODE_DIR} not found. Clone https://github.com/inclusionAI/Ming there first "
                "(see module docstring) -- the custom model classes aren't resolved via "
                "AutoModel(trust_remote_code=True) in any upstream example."
            )
        sys.path.insert(0, str(MING_CODE_DIR))

    import torch
    from transformers import AutoProcessor
    from modeling_bailingmm2 import BailingMM2NativeForConditionalGeneration

    num_gpus = torch.cuda.device_count()
    device_map = _split_model(num_gpus) if (load_image_gen or load_talker) else "auto"

    _model = BailingMM2NativeForConditionalGeneration.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map=device_map,
        load_image_gen=load_image_gen,
        load_talker=load_talker,
    ).to(dtype=torch.bfloat16)
    _processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    return _model, _processor


def generate_story(max_new_tokens: int = 512) -> str:
    """One narration+dialogue beat for the scene, in the same spirit as the
    Director's job in director/generate.py -- but asking the omni model
    itself, to see whether its native text generation is competitive."""
    import torch

    model, processor = _load(load_image_gen=False, load_talker=False)
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
    """Unified character+background still, in the spirit of
    illustration/generate.py -- but from Ming-Flash-Omni's own image_gen head
    instead of Illustrious-XL, to compare quality/coherence directly."""
    model, processor = _load(load_image_gen=True, load_talker=False)
    messages = [{"role": "HUMAN", "content": [{"type": "text", "text": illustration_prompt}]}]

    text = processor.apply_chat_template(messages, add_generation_prompt=True)
    image_inputs, _, _ = processor.process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, return_tensors="pt").to(model.device)

    image = model.generate(**inputs, image_gen=True, image_gen_seed=seed)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path)
    return out_path


def generate_narration_with_bgm(
    narration_text: str, bgm_mood: str, bgm_instrument: str, out_path: Path,
) -> Path:
    """Narrated speech with a BGM bed under it, via the talker's
    instruct_audio_generation + a BGM instruction field. NOT a standalone
    instrumental BGM generator (see module docstring) -- this tests how good
    the BGM component sounds underneath narration, as a proxy for whether
    the joint generation buys any mood/style coherence over calling
    MusicGen independently."""
    import torch
    import torchaudio

    model, _ = _load(load_image_gen=False, load_talker=True)
    instruction = json.dumps(
        {"audio_sequence": [{"BGM": {"Genre": None, "Mood": bgm_mood, "Instrument": bgm_instrument,
                                      "Theme": None, "ENV": None, "SNR": None}}]},
        ensure_ascii=False,
    )

    all_wavs = []
    for tts_speech, _, _, _ in model.talker.instruct_audio_generation(
        prompt="Please generate speech based on the following description.\n",
        text=narration_text,
        use_zero_spk_emb=True,
        instruction=instruction,
        max_decode_steps=200,
        cfg=2.0, sigma=0.25, temperature=0,
        max_length=50,
        audio_detokenizer=model.talker_vae,
        stream=True,
        taskname="TTS",
    ):
        all_wavs.append(tts_speech)

    waveform = torch.cat(all_wavs, dim=-1)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(out_path), waveform, sample_rate=model.talker_vae.config.sample_rate)
    return out_path


def run(dry_run: bool = False) -> None:
    scene = SAMPLE_SCENE
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if dry_run:
        print("[dry-run] would generate story for scene:", json.dumps(scene, ensure_ascii=False, indent=2))
        print("[dry-run] would generate illustration for prompt:", ILLUSTRATION_PROMPT)
        print("[dry-run] would generate narration+BGM with mood:", scene["bgm_mood"], scene["bgm_instrument"])
        print(f"[dry-run] outputs would be written under {OUT_DIR}")
        return

    t0 = time.time()
    print("[1/3] story...", file=sys.stderr)
    story = generate_story()
    (OUT_DIR / "story.txt").write_text(story, encoding="utf-8")
    print(story)
    print(f"  ({time.time() - t0:.1f}s)", file=sys.stderr)

    t0 = time.time()
    print("[2/3] illustration...", file=sys.stderr)
    illustration_path = generate_illustration(ILLUSTRATION_PROMPT, OUT_DIR / "illustration.jpg")
    print(f"  wrote {illustration_path} ({time.time() - t0:.1f}s)", file=sys.stderr)

    t0 = time.time()
    print("[3/3] narration + BGM...", file=sys.stderr)
    # Use the story's first line as the narration text fed to the talker.
    narration_line = story.strip().splitlines()[0] if story.strip() else scene["context"]
    audio_path = generate_narration_with_bgm(
        narration_line, scene["bgm_mood"], scene["bgm_instrument"], OUT_DIR / "narration_bgm.wav",
    )
    print(f"  wrote {audio_path} ({time.time() - t0:.1f}s)", file=sys.stderr)


def _cli():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print the scene/prompts that would be sent without loading the model",
    )
    args = parser.parse_args()
    run(dry_run=args.dry_run)


if __name__ == "__main__":
    _cli()
