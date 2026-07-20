"""Manual kohya-format LoRA merging for the SDXL text encoders.

diffusers' built-in `load_lora_weights` uses a kohya->diffusers key-translation
table that, at least in diffusers 0.39.0, produces a broken/legacy-format state
dict for some community LoRAs (fails with `IndexError` deep in peft_utils).
Merging weights directly, by matching each LoRA module's kohya-style flattened
name against the pipeline's real `named_modules()` output, sidesteps that
entirely for the text encoders.

This intentionally does NOT cover the UNet: kohya trains against the original
Stability UNet naming ("input_blocks.N", "middle_block", "output_blocks.N"),
which does not match diffusers' UNet2DConditionModel naming ("down_blocks",
"mid_block", "up_blocks") — that requires a real block-index translation table,
not a rename. In practice the text-encoder-only merge already carries most of
a style LoRA's effect for prompt-driven style transfer, since it's what shifts
cross-attention conditioning.

Merging is done in place and is NOT reversible without reloading the pipeline
from scratch — fine for this project's one-shot-per-process usage pattern.
"""
import torch
from safetensors.torch import load_file


def merge_text_encoder_lora(pipe, lora_path, multiplier: float = 1.2) -> int:
    """Merge a kohya-format LoRA's text-encoder weights into pipe.text_encoder(_2)
    in place. Returns the number of modules merged (0 means the LoRA had no
    text-encoder layers, or its naming didn't match — check before relying on it)."""
    sd = load_file(str(lora_path))
    submodels = {
        "lora_te1": pipe.text_encoder,
        "lora_te2": pipe.text_encoder_2,
    }
    applied = 0
    for prefix, submodel in submodels.items():
        for name, module in submodel.named_modules():
            if not hasattr(module, "weight") or module.weight is None:
                continue
            flat = name.replace(".", "_")
            candidates = (f"{prefix}_{flat}", f"{prefix}_text_model_{flat}")
            key = next((c for c in candidates if f"{c}.lora_down.weight" in sd), None)
            if key is None:
                continue
            down = sd[f"{key}.lora_down.weight"].to(torch.float32)
            up = sd[f"{key}.lora_up.weight"].to(torch.float32)
            rank = down.shape[0]
            alpha = sd[f"{key}.alpha"].item() if f"{key}.alpha" in sd else float(rank)
            scale = alpha / rank
            delta = (up.reshape(up.shape[0], -1) @ down.reshape(down.shape[0], -1)).reshape(module.weight.shape)
            module.weight.data += multiplier * scale * delta.to(module.weight.device, module.weight.dtype)
            applied += 1
    return applied
