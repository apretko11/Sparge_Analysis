import argparse
import json
import os
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def cuda_event_time_ms(fn, warmup=2, repeats=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    times = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        fn()
        end.record()

        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    return times


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--seq-len", type=int, default=8192)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--use-cache", action="store_true")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    dtype_map = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }
    dtype = dtype_map[args.dtype]
    device = "cuda"

    print(f"Loading model: {args.model_dir}")
    print(f"Label: {args.label}")
    print(f"seq_len={args.seq_len}, batch_size={args.batch_size}, dtype={args.dtype}, use_cache={args.use_cache}")
    print(f"CELERITY_USE_SPARGE_ATTN={os.environ.get('CELERITY_USE_SPARGE_ATTN')}")
    print(f"CELERITY_SPARGE_IGNORE_BIAS={os.environ.get('CELERITY_SPARGE_IGNORE_BIAS')}")
    print(f"CELERITY_SPARGE_IGNORE_MASK={os.environ.get('CELERITY_SPARGE_IGNORE_MASK')}")
    print(f"CELERITY_SPARGE_TOPK={os.environ.get('CELERITY_SPARGE_TOPK')}")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_dir,
        trust_remote_code=True,
        fix_mistral_regex=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        trust_remote_code=True,
        torch_dtype=dtype,
    ).to(device).eval()

    text_unit = "The quick brown fox jumps over the lazy dog. "
    prompt = text_unit

    while True:
        prompt += text_unit * 100
        n_tokens = len(tokenizer(prompt).input_ids)
        if n_tokens >= args.seq_len:
            break

    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=args.seq_len,
    )

    inputs = {k: v.to(device) for k, v in encoded.items()}

    if args.batch_size != 1:
        inputs = {k: v.repeat(args.batch_size, 1) for k, v in inputs.items()}

    actual_seq_len = inputs["input_ids"].shape[1]
    print(f"Prompt length: {actual_seq_len:,} tokens")

    with torch.no_grad():
        out = model(**inputs, use_cache=args.use_cache)
        assert torch.isfinite(out.logits).all().item(), "Non-finite logits detected"

    def fn():
        with torch.no_grad():
            _ = model(**inputs, use_cache=args.use_cache)

    times_ms = cuda_event_time_ms(fn, warmup=args.warmup, repeats=args.repeats)

    result = {
        "label": args.label,
        "model_dir": args.model_dir,
        "prompt_mode": "repeated_text",
        "text_unit": text_unit,
        "target_seq_len": args.seq_len,
        "actual_seq_len": actual_seq_len,
        "batch_size": args.batch_size,
        "dtype": args.dtype,
        "use_cache": args.use_cache,
        "times_ms": times_ms,
        "mean_ms": sum(times_ms) / len(times_ms),
        "min_ms": min(times_ms),
        "max_ms": max(times_ms),
        "tokens_per_sec_prefill": actual_seq_len / ((sum(times_ms) / len(times_ms)) / 1000.0),
        "cuda_device": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "env": {
            "CELERITY_USE_SPARGE_ATTN": os.environ.get("CELERITY_USE_SPARGE_ATTN"),
            "CELERITY_SPARGE_IGNORE_BIAS": os.environ.get("CELERITY_SPARGE_IGNORE_BIAS"),
            "CELERITY_SPARGE_IGNORE_MASK": os.environ.get("CELERITY_SPARGE_IGNORE_MASK"),
            "CELERITY_SPARGE_TOPK": os.environ.get("CELERITY_SPARGE_TOPK"),
        },
    }

    print(json.dumps(result, indent=2))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
