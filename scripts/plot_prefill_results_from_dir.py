import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


SIZES = ["h640", "h896", "h1152"]
SEQS = [1024, 4096, 8192]


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def find_one(results_dir, pattern):
    matches = sorted(results_dir.glob(pattern))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected exactly one match for {pattern}, found {len(matches)}: {matches}")
    return matches[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--plots-dir", required=True)
    parser.add_argument("--title-suffix", default="")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    plots_dir = Path(args.plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    for size in SIZES:
        baseline_means = []
        sparge_means = []
        speedups = []

        for seq in SEQS:
            baseline_path = find_one(results_dir, f"{size}_baseline_seq{seq}*.json")
            sparge_path = find_one(results_dir, f"{size}_sparge_seq{seq}*.json")

            baseline = load_json(baseline_path)
            sparge = load_json(sparge_path)

            baseline_ms = float(baseline["mean_ms"])
            sparge_ms = float(sparge["mean_ms"])
            speedup = baseline_ms / sparge_ms

            baseline_means.append(baseline_ms)
            sparge_means.append(sparge_ms)
            speedups.append(speedup)

            rows.append({
                "size": size,
                "seq_len": seq,
                "baseline_file": baseline_path.name,
                "sparge_file": sparge_path.name,
                "baseline_mean_ms": baseline_ms,
                "sparge_mean_ms": sparge_ms,
                "speedup": speedup,
                "baseline_min_ms": float(baseline["min_ms"]),
                "baseline_max_ms": float(baseline["max_ms"]),
                "sparge_min_ms": float(sparge["min_ms"]),
                "sparge_max_ms": float(sparge["max_ms"]),
                "baseline_use_cache": baseline.get("use_cache"),
                "sparge_use_cache": sparge.get("use_cache"),
                "baseline_prompt_mode": baseline.get("prompt_mode", "synthetic_random_tokens"),
                "sparge_prompt_mode": sparge.get("prompt_mode", "synthetic_random_tokens"),
            })

        x = np.arange(len(SEQS))
        width = 0.36

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(x - width / 2, baseline_means, width, label="Baseline")
        ax.bar(x + width / 2, sparge_means, width, label="SpargeAttention")

        title_suffix = f", {args.title_suffix}" if args.title_suffix else ""
        ax.set_title(f"{size} prefill latency{title_suffix}")
        ax.set_xlabel("Sequence length")
        ax.set_ylabel("Mean latency (ms)")
        ax.set_xticks(x)
        ax.set_xticklabels([str(s) for s in SEQS])
        ax.legend()

        for i, speedup in enumerate(speedups):
            ymax = max(baseline_means[i], sparge_means[i])
            ax.text(
                x[i],
                ymax * 1.03,
                f"{speedup:.2f}×",
                ha="center",
                va="bottom",
                fontsize=10,
            )

        fig.tight_layout()
        out_path = plots_dir / f"plot_{size}_prefill.png"
        fig.savefig(out_path, dpi=200)
        plt.close(fig)

        print(f"Wrote {out_path}")

    csv_path = plots_dir / "results_summary.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "size",
                "seq_len",
                "baseline_file",
                "sparge_file",
                "baseline_mean_ms",
                "sparge_mean_ms",
                "speedup",
                "baseline_min_ms",
                "baseline_max_ms",
                "sparge_min_ms",
                "sparge_max_ms",
                "baseline_use_cache",
                "sparge_use_cache",
                "baseline_prompt_mode",
                "sparge_prompt_mode",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {csv_path}")

    print("\nSummary:")
    for row in rows:
        print(
            f"{row['size']} seq{row['seq_len']}: "
            f"baseline={row['baseline_mean_ms']:.2f} ms, "
            f"sparge={row['sparge_mean_ms']:.2f} ms, "
            f"speedup={row['speedup']:.2f}x"
        )


if __name__ == "__main__":
    main()
