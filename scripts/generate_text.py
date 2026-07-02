#!/usr/bin/env python3

import argparse
import json
import os
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_tokenizer(model_name: str, trust_remote_code: bool):
    try:
        return AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
            fix_mistral_regex=True,
        )
    except TypeError:
        return AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
        )


def main():
    parser = argparse.ArgumentParser(
        description="Generate text from a Hugging Face causal language model."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="models/baseline/h640",
        help="Hugging Face model name or local model path.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Once upon a time",
        help="Input prompt.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=100,
        help="Number of new tokens to generate.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature. Used only with --do-sample.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.95,
        help="Top-p sampling value. Used only with --do-sample.",
    )
    parser.add_argument(
        "--do-sample",
        action="store_true",
        help="Enable sampling. Default is greedy decoding.",
    )
    parser.add_argument(
        "--dtype",
        choices=["auto", "bf16", "fp16", "fp32"],
        default="auto",
        help="Model dtype.",
    )
    parser.add_argument(
        "--device-map",
        type=str,
        default="auto",
        help='Device map passed to from_pretrained. Use "auto" or "none".',
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        default=True,
        help="Trust custom model code. Default: true.",
    )
    parser.add_argument(
        "--no-trust-remote-code",
        dest="trust_remote_code",
        action="store_false",
        help="Disable trust_remote_code.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Random seed.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Optional JSON output path.",
    )

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    dtype_map = {
        "auto": "auto",
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }
    torch_dtype = dtype_map[args.dtype]
    device_map = None if args.device_map == "none" else args.device_map

    print(f"Loading model: {args.model}")
    print(f"dtype={args.dtype}, device_map={args.device_map}")
    print(f"trust_remote_code={args.trust_remote_code}")
    print(f"CELERITY_USE_SPARGE_ATTN={os.environ.get('CELERITY_USE_SPARGE_ATTN')}")
    print(f"CELERITY_SPARGE_IGNORE_BIAS={os.environ.get('CELERITY_SPARGE_IGNORE_BIAS')}")
    print(f"CELERITY_SPARGE_IGNORE_MASK={os.environ.get('CELERITY_SPARGE_IGNORE_MASK')}")
    print(f"CELERITY_SPARGE_TOPK={os.environ.get('CELERITY_SPARGE_TOPK')}")

    tokenizer = load_tokenizer(args.model, args.trust_remote_code)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch_dtype,
        device_map=device_map,
        trust_remote_code=args.trust_remote_code,
    )
    model.eval()

    inputs = tokenizer(args.prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    prompt_tokens = inputs["input_ids"].shape[1]

    generate_kwargs = {
        **inputs,
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.do_sample,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "use_cache": True,
    }

    if args.do_sample:
        generate_kwargs["temperature"] = args.temperature
        generate_kwargs["top_p"] = args.top_p

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    start_time = time.perf_counter()

    with torch.no_grad():
        outputs = model.generate(**generate_kwargs)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - start_time

    generated_ids = outputs[0, prompt_tokens:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    new_tokens = generated_ids.shape[0]
    tok_per_s = new_tokens / elapsed if elapsed > 0 else None

    print("\n=== Prompt ===\n")
    print(args.prompt)

    print("\n=== Generated Text ===\n")
    print(generated_text)

    print("\n=== Full Text ===\n")
    print(full_text)

    print("\n=== Stats ===")
    print(f"Prompt tokens: {prompt_tokens}")
    print(f"Generated tokens: {new_tokens}")
    print(f"Generation time: {elapsed:.4f} s")
    print(f"Generated tokens/sec: {tok_per_s:.2f}")

    result = {
        "model": args.model,
        "prompt": args.prompt,
        "prompt_tokens": int(prompt_tokens),
        "max_new_tokens": args.max_new_tokens,
        "generated_tokens": int(new_tokens),
        "generation_time_s": elapsed,
        "generated_tokens_per_sec": tok_per_s,
        "do_sample": args.do_sample,
        "temperature": args.temperature if args.do_sample else None,
        "top_p": args.top_p if args.do_sample else None,
        "dtype": args.dtype,
        "device_map": args.device_map,
        "generated_text": generated_text,
        "full_text": full_text,
        "torch_version": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "env": {
            "CELERITY_USE_SPARGE_ATTN": os.environ.get("CELERITY_USE_SPARGE_ATTN"),
            "CELERITY_SPARGE_IGNORE_BIAS": os.environ.get("CELERITY_SPARGE_IGNORE_BIAS"),
            "CELERITY_SPARGE_IGNORE_MASK": os.environ.get("CELERITY_SPARGE_IGNORE_MASK"),
            "CELERITY_SPARGE_TOPK": os.environ.get("CELERITY_SPARGE_TOPK"),
        },
    }

    if args.out is not None:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2))
        print(f"\nWrote: {out_path}")


if __name__ == "__main__":
    main()
