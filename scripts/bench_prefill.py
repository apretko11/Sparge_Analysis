import argparse
import json
import os
import time
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM


def cuda_event_time_ms(fn, warmup=3, repeats=10):
    # Warmup
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
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
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

    model_dir = args.model_dir
    device = "cuda"

    print(f"Loading model: {model_dir}")
    print(f"Label: {args.label}")
    print(f"seq_len={args.seq_len}, batch_size={args.batch_size}, dtype={args.dtype}, use_cache={args.use_cache}")
    print(f"CELERITY_USE_SPARGE_ATTN={os.environ.get('CELERITY_USE_SPARGE_ATTN')}")
    print(f"CELERITY_SPARGE_IGNORE_BIAS={os.environ.get('CELERITY_SPARGE_IGNORE_BIAS')}")
    print(f"CELERITY_SPARGE_IGNORE_MASK={os.environ.get('CELERITY_SPARGE_IGNORE_MASK')}")
    print(f"CELERITY_SPARGE_TOPK={os.environ.get('CELERITY_SPARGE_TOPK')}")

    config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    vocab_size = int(config.vocab_size)

    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        trust_remote_code=True,
        torch_dtype=dtype,
    ).to(device).eval()

    # Synthetic input IDs avoid timing tokenization and give exact seq_len.
    # Keep IDs inside the normal vocab range.
    torch.manual_seed(1)
    input_ids = torch.randint(
        low=0,
        high=vocab_size,
        size=(args.batch_size, args.seq_len),
        device=device,
        dtype=torch.long,
    )

    # Sanity forward
    with torch.no_grad():
        out = model(input_ids=input_ids, use_cache=args.use_cache)
        assert torch.isfinite(out.logits).all().item(), "Non-finite logits detected"

    def fn():
        with torch.no_grad():
            _ = model(input_ids=input_ids, use_cache=args.use_cache)

    times_ms = cuda_event_time_ms(fn, warmup=args.warmup, repeats=args.repeats)

    result = {
        "label": args.label,
        "model_dir": str(model_dir),
        "seq_len": args.seq_len,
        "batch_size": args.batch_size,
        "dtype": args.dtype,
        "use_cache": args.use_cache,
        "times_ms": times_ms,
        "mean_ms": sum(times_ms) / len(times_ms),
        "min_ms": min(times_ms),
        "max_ms": max(times_ms),
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
