"""Manual kohya-format LoRA merging for Anima's CosmosTransformer3DModel.

Same situation as lora_merge.py's SDXL case: the LoRA was trained against the
original (non-diffusers) Cosmos block naming ("blocks.N.self_attn.q_proj",
"blocks.N.mlp.layer1", ...) while diffusers' CosmosTransformer3DModel uses
different names ("transformer_blocks.N.attn1.to_q",
"transformer_blocks.N.ff.net.0.proj", ...) — diffusers' own kohya conversion
utilities don't know this specific model, so there's no automatic path.
Unlike the SDXL text-encoder case, this LoRA targets the diffusion transformer
itself (where most of a style's visual character lives), not just a text
encoder, so this merge should carry the LoRA's full intended effect rather
than a partial one.

Merging is done in place and is NOT reversible without reloading the model —
fine for this project's one-shot-per-process usage pattern.
"""
import re

import torch
from safetensors.torch import load_file

_ATTN_PROJ_MAP = {
    "self_attn_q_proj": "attn1.to_q",
    "self_attn_k_proj": "attn1.to_k",
    "self_attn_v_proj": "attn1.to_v",
    "self_attn_output_proj": "attn1.to_out.0",
    "cross_attn_q_proj": "attn2.to_q",
    "cross_attn_k_proj": "attn2.to_k",
    "cross_attn_v_proj": "attn2.to_v",
    "cross_attn_output_proj": "attn2.to_out.0",
    "mlp_layer1": "ff.net.0.proj",
    "mlp_layer2": "ff.net.2",
}
_BLOCK_KEY_RE = re.compile(r"^lora_unet_blocks_(\d+)_(.+)$")


def merge_cosmos_transformer_lora(transformer, lora_path, multiplier: float = 1.0) -> int:
    """Merge a kohya-format LoRA's transformer weights into `transformer` in
    place. Returns the number of modules merged (0 means nothing matched —
    check before relying on it)."""
    sd = load_file(str(lora_path))
    down_keys = [k for k in sd if k.endswith(".lora_down.weight")]

    applied = 0
    for down_key in down_keys:
        base = down_key[: -len(".lora_down.weight")]
        match = _BLOCK_KEY_RE.match(base)
        if not match:
            continue
        block_idx, suffix = match.groups()
        target_suffix = _ATTN_PROJ_MAP.get(suffix)
        if target_suffix is None:
            continue

        module_name = f"transformer_blocks.{block_idx}.{target_suffix}"
        try:
            module = transformer.get_submodule(module_name)
        except AttributeError:
            continue

        down = sd[f"{base}.lora_down.weight"].to(torch.float32)
        up = sd[f"{base}.lora_up.weight"].to(torch.float32)
        rank = down.shape[0]
        alpha = sd[f"{base}.alpha"].item() if f"{base}.alpha" in sd else float(rank)
        scale = alpha / rank
        delta = (up.reshape(up.shape[0], -1) @ down.reshape(down.shape[0], -1)).reshape(module.weight.shape)
        module.weight.data += multiplier * scale * delta.to(module.weight.device, module.weight.dtype)
        applied += 1

    return applied
