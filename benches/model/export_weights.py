#!/usr/bin/env python3
"""Export a Llama checkpoint (Hugging Face safetensors) as the raw files examples/llama.mapal reads.

`read` (ADR-0045) loads a value's bytes: raw little-endian elements, nothing else. The program reads the whole model
as one value of its `Model` type, so `model.bin` is that value's byte image, fields in declaration order:

    embedding          [vocabulary_size x hidden_size]                 bf16 (tied: also the output head)
    layers[0 .. L-1]   each a DecoderLayerWeights, fields in order:
        attention_norm     [hidden_size]                               bf16
        query              [query_heads * head_size x hidden_size]     bf16
        key                [key_value_heads * head_size x hidden_size] bf16
        value              [key_value_heads * head_size x hidden_size] bf16
        output             [hidden_size x query_heads * head_size]     bf16
        feed_forward_norm  [hidden_size]                               bf16
        gate               [feed_forward_size x hidden_size]           bf16
        up                 [feed_forward_size x hidden_size]           bf16
        down               [hidden_size x feed_forward_size]           bf16
    final_norm         [hidden_size]                                   bf16

Matrices are the checkpoint's own (out rows of in weights, row-major), copied bit for bit. `prompt.bin` is the
prompt token ids (i32): row 0 of the competitor harness's longest prompt; the shorter prompts are its prefixes.

Usage:  python export_weights.py [--model DIR] [--out DIR] [--prompt FILE.i32 --prompt-length N]
Defaults: ~/.cache/mapal/models/llama-3.2-1b-instruct -> <model>/mapal/ (outside the repository).
"""
import argparse
import json
import os

import torch
from safetensors import safe_open

LAYER_FIELDS = [
    ("attention_norm", "input_layernorm.weight"),
    ("query", "self_attn.q_proj.weight"),
    ("key", "self_attn.k_proj.weight"),
    ("value", "self_attn.v_proj.weight"),
    ("output", "self_attn.o_proj.weight"),
    ("feed_forward_norm", "post_attention_layernorm.weight"),
    ("gate", "mlp.gate_proj.weight"),
    ("up", "mlp.up_proj.weight"),
    ("down", "mlp.down_proj.weight"),
]


def raw_bf16(t):
    assert t.dtype == torch.bfloat16, t.dtype
    return t.contiguous().view(torch.int16).numpy().tobytes()


def tensor_files(model_dir):
    index = os.path.join(model_dir, "model.safetensors.index.json")
    if os.path.exists(index):
        names = sorted(set(json.load(open(index))["weight_map"].values()))
        return [os.path.join(model_dir, n) for n in names]
    return [os.path.join(model_dir, "model.safetensors")]


def main():
    home = os.path.expanduser("~/.cache/mapal")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=os.path.join(home, "models", "llama-3.2-1b-instruct"))
    ap.add_argument("--out", default=None, help="default: <model>/mapal")
    ap.add_argument("--prompt", default=os.path.join(home, "model-compare", "prompts", "prompt_2048.i32"))
    ap.add_argument("--prompt-length", type=int, default=2048, help="ids per row of --prompt")
    args = ap.parse_args()
    out = args.out or os.path.join(args.model, "mapal")
    os.makedirs(out, exist_ok=True)

    config = json.load(open(os.path.join(args.model, "config.json")))
    layers = config["num_hidden_layers"]
    handles = [safe_open(f, "pt") for f in tensor_files(args.model)]

    def get(name):
        for h in handles:
            if name in h.keys():
                return h.get_tensor(name)
        raise KeyError(name)

    written = 0
    with open(os.path.join(out, "model.bin"), "wb") as f:
        written += f.write(raw_bf16(get("model.embed_tokens.weight")))
        for layer in range(layers):
            for _, suffix in LAYER_FIELDS:
                written += f.write(raw_bf16(get(f"model.layers.{layer}.{suffix}")))
        written += f.write(raw_bf16(get("model.norm.weight")))
    if not config.get("tie_word_embeddings", False):
        raise SystemExit("the program ties the output head to the embedding; this checkpoint has its own lm_head")

    ids = torch.frombuffer(bytearray(open(args.prompt, "rb").read()), dtype=torch.int32)
    row = ids[: args.prompt_length]
    with open(os.path.join(out, "prompt.bin"), "wb") as f:
        f.write(row.numpy().tobytes())

    sizes = {k: config.get(k) for k in ("vocab_size", "hidden_size", "num_hidden_layers", "num_attention_heads",
                                        "num_key_value_heads", "intermediate_size", "rms_norm_eps", "rope_theta")}
    sizes["head_dim"] = config.get("head_dim", config["hidden_size"] // config["num_attention_heads"])
    sizes["rope_scaling"] = config.get("rope_scaling") or config.get("rope_parameters")
    sizes["model_bin_bytes"] = written
    sizes["prompt_ids"] = int(row.numel())
    json.dump(sizes, open(os.path.join(out, "sizes.json"), "w"), indent=1)
    print(f"{out}/model.bin: {written} bytes ({layers} layers); prompt.bin: {row.numel()} ids")


if __name__ == "__main__":
    main()
