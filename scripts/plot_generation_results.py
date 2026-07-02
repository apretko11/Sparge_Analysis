#!/usr/bin/env python3

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


SIZES = ["h640", "h896", "h1152"]
VARIANTS = ["baseline", "sparge"]
PROMPT_TYPES = ["short", "long"]


def result_path(size, variant, prompt_type):
    if prompt_type == "short":
        return Path(f"results/generation_{size}_{variant}.json")
    if prompt_type == "long":
        return Path(f"results/generation_{size}_{variant}_long_prompt.json")
    raise ValueError(f"Unknown prompt_type: {prompt_type}")


def load_result(size, variant, prompt_type):
    path = result_path(size, variant, prompt_type)
    if not path.exists():
        raise FileNotFoundError(path)

    with open(path, "r") as f:
        data = json.load(f)

    return {
        "size": size,
        "variant": variant,
        "prompt_type": prompt_type,
        "file": str(path),
        "model": data.get("model"),
        "prompt_tokens": data.get("prompt_tokens"),
        "generated_tokens": data.get("generated_tokens"),
        "generation_time_s": data.get("generation_time_s"),
        "generated_tokens_per_sec": data.get("generated_tokens_per_sec"),
        "dtype": data.get("dtype"),
        "device_map": data.get("device_map"),
        "cuda_device": data.get("cuda_device"),
        "celerity_use_sparge_attn": data.get("env", {}).get("CELERITY_USE_SPARGE_ATTN"),
        "celerity_sparge_topk": data.get("env", {}).get("CELERITY_SPARGE_TOPK"),
    }


def make_grouped_plot(rows, prompt_type, metric_key, ylabel, title, out_path):
    prompt_rows = [r for r in rows if r["prompt_type"] == prompt_type]

    baseline_values = []
    sparge_values = []
    ratios = []

    for size in SIZES:
        baseline = next(
            r for r in prompt_rows
            if r["size"] == size and r["variant"] == "baseline"
        )
        sparge = next(
            r for r in prompt_rows
            if r["size"] == size and r["variant"] == "sparge"
        )

        b = float(baseline[metric_key])
        s = float(sparge[metric_key])

        baseline_values.append(b)
        sparge_values.append(s)

        if metric_key == "generation_time_s":
            ratios.append(b / s)
        else:
            ratios.append(s / b)

    x = np.arange(len(SIZES))
    width = 0.36

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width / 2, baseline_values, width, label="Baseline")
    ax.bar(x + width / 2, sparge_values, width, label="Sparge")

    ax.set_title(title)
    ax.set_xlabel("Model size")
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(SIZES)
    ax.legend()

    for i, ratio in enumerate(ratios):
        ymax = max(baseline_values[i], sparge_values[i])
        if metric_key == "generation_time_s":
            label = f"{ratio:.2f}× time ratio"
        else:
            label = f"{ratio:.2f}× tok/s ratio"

        ax.text(
            x[i],
            ymax * 1.03,
            label,
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Wrote {out_path}")


def main():
    plots_dir = Path("plots/generation")
    plots_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for size in SIZES:
        for prompt_type in PROMPT_TYPES:
            for variant in VARIANTS:
                rows.append(load_result(size, variant, prompt_type))

    csv_path = plots_dir / "results_summary.csv"
    fieldnames = [
        "size",
        "variant",
        "prompt_type",
        "file",
        "model",
        "prompt_tokens",
        "generated_tokens",
        "generation_time_s",
        "generated_tokens_per_sec",
        "dtype",
        "device_map",
        "cuda_device",
        "celerity_use_sparge_attn",
        "celerity_sparge_topk",
    ]

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {csv_path}")

    make_grouped_plot(
        rows,
        prompt_type="short",
        metric_key="generation_time_s",
        ylabel="Generation time (s)",
        title="Generation smoke test, short prompt: elapsed time",
        out_path=plots_dir / "generation_short_prompt_elapsed_time.png",
    )

    make_grouped_plot(
        rows,
        prompt_type="short",
        metric_key="generated_tokens_per_sec",
        ylabel="Generated tokens/sec",
        title="Generation smoke test, short prompt: generated token throughput",
        out_path=plots_dir / "generation_short_prompt_tokens_per_sec.png",
    )

    make_grouped_plot(
        rows,
        prompt_type="long",
        metric_key="generation_time_s",
        ylabel="Generation time (s)",
        title="Generation smoke test, long prompt: elapsed time",
        out_path=plots_dir / "generation_long_prompt_elapsed_time.png",
    )

    make_grouped_plot(
        rows,
        prompt_type="long",
        metric_key="generated_tokens_per_sec",
        ylabel="Generated tokens/sec",
        title="Generation smoke test, long prompt: generated token throughput",
        out_path=plots_dir / "generation_long_prompt_tokens_per_sec.png",
    )

    print("\nSummary:")
    for row in rows:
        print(
            f"{row['size']} {row['variant']} {row['prompt_type']}: "
            f"prompt_tokens={row['prompt_tokens']}, "
            f"generated_tokens={row['generated_tokens']}, "
            f"time={row['generation_time_s']:.4f}s, "
            f"tok/s={row['generated_tokens_per_sec']:.2f}"
        )


if __name__ == "__main__":
    main()
