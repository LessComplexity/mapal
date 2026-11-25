#!/usr/bin/env python3
"""Greedy reference tokens for examples/llama.mapal from Hugging Face transformers.

The same checkpoint, prompt ids and greedy rule (argmax, first index on a tie, no end-of-text stop):
  --math f32   bf16 weights widened to f32, every activation f32, eager attention (Mapal's default face:
               f32 math on the bf16 weights)
  --math bf16  the checkpoint as shipped: bf16 weights and activations, eager attention
Writes JSON {tokens, top_logits, math, device} and the first step's logits (<out>.logits0.npy) so a run's
first-step logits can be compared value by value.

Usage: python hf_reference.py --n 512 --gen 128 --math f32 --out hf_f32_n512.json
"""
import argparse
import json
import os

import numpy as np
import torch
from transformers import AutoModelForCausalLM


def main():
    home = os.path.expanduser("~/.cache/mapal")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=os.path.join(home, "models", "llama-3.2-1b-instruct"))
    ap.add_argument("--prompt", default=None, help="i32 ids (default: <model>/mapal/prompt.bin)")
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--gen", type=int, default=128)
    ap.add_argument("--math", choices=["f32", "bf16"], default="f32")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    prompt = args.prompt or os.path.join(args.model, "mapal", "prompt.bin")
    ids = np.fromfile(prompt, dtype=np.int32)[: args.n].astype(np.int64)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, attn_implementation="eager")
    if args.math == "f32":
        model = model.float()
    model = model.to(args.device).eval()

    tokens, top = [], []
    with torch.no_grad():
        x = torch.tensor(ids, device=args.device)[None, :]
        past = None
        for step in range(args.gen):
            out = model(input_ids=x, past_key_values=past, use_cache=True)
            past = out.past_key_values
            logits = out.logits[0, -1].float()
            if step == 0:
                np.save(args.out + ".logits0.npy", logits.cpu().numpy())
            t = int(torch.argmax(logits))
            tokens.append(t)
            top.append(float(logits[t]))
            x = torch.tensor([[t]], device=args.device)
    json.dump({"tokens": tokens, "top_logits": top, "math": args.math, "device": args.device, "n": args.n},
              open(args.out, "w"))
    print(" ".join(map(str, tokens)))


if __name__ == "__main__":
    main()
