#!/usr/bin/env python3
"""Sweep block/window lengths for ACTW and LZ entropy estimators."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev

import random

from entropy_estimators import (
    ctw_entropy,
    fixed_block_actw_entropy,
    lz_gkb_entropy,
    make_sources,
    random_window_actw_entropy,
)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_sensitivity_svg(path: Path, rows: list[dict], metric: str, title: str) -> None:
    families = ["stationary", "nonstationary"]
    estimators = sorted({row["estimator"] for row in rows})
    block_lengths = sorted({int(row["block_length"]) for row in rows})
    grouped: dict[tuple[str, str, int], list[float]] = defaultdict(list)
    for row in rows:
        grouped[(row["family"], row["estimator"], int(row["block_length"]))].append(float(row[metric]))

    width, height = 1200, 520
    margin_left, margin_right, margin_top, margin_bottom = 80, 30, 70, 90
    panel_gap = 50
    panel_width = (width - margin_left - margin_right - panel_gap) / 2
    plot_height = height - margin_top - margin_bottom
    values = [float(row[metric]) for row in rows]
    y_min, y_max = min(values), max(values)
    if metric == "bias":
        bound = max(abs(y_min), abs(y_max), 0.01)
        y_min, y_max = -bound, bound
    elif metric == "mean_estimate":
        y_min = 0.0
    if y_min == y_max:
        y_min -= 0.05
        y_max += 0.05

    colors = {"CTW": "#4C78A8", "ACTW_fixed": "#F58518", "ACTW_random": "#54A24B", "LZ_GKB": "#E45756"}

    def x_pos(panel: int, idx: int) -> float:
        x0 = margin_left + panel * (panel_width + panel_gap)
        return x0 + (idx + 0.5) * panel_width / len(block_lengths)

    def y_pos(v: float) -> float:
        return margin_top + (y_max - v) / (y_max - y_min) * plot_height

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style>text{font-family:Arial,sans-serif;font-size:12px}.title{font-size:20px;font-weight:700}.panel{font-size:15px;font-weight:700}</style>",
        f'<text x="{width/2:.1f}" y="30" text-anchor="middle" class="title">{title}</text>',
    ]

    for panel, family in enumerate(families):
        x0 = margin_left + panel * (panel_width + panel_gap)
        x1 = x0 + panel_width
        lines.append(f'<line x1="{x0:.1f}" y1="{margin_top}" x2="{x0:.1f}" y2="{margin_top+plot_height}" stroke="#333"/>')
        lines.append(f'<line x1="{x0:.1f}" y1="{margin_top+plot_height}" x2="{x1:.1f}" y2="{margin_top+plot_height}" stroke="#333"/>')
        lines.append(f'<text x="{(x0+x1)/2:.1f}" y="55" text-anchor="middle" class="panel">{family}</text>')
        for bi, block in enumerate(block_lengths):
            x = x_pos(panel, bi)
            lines.append(f'<text x="{x:.1f}" y="{height-20}" text-anchor="middle">{block}</text>')
        for estimator in estimators:
            points = []
            for bi, block in enumerate(block_lengths):
                vals = grouped.get((family, estimator, block), [])
                if not vals:
                    continue
                points.append((x_pos(panel, bi), y_pos(mean(vals))))
            if len(points) < 2:
                continue
            color = colors.get(estimator, "#666")
            poly = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
            lines.append(f'<polyline fill="none" stroke="{color}" stroke-width="2" points="{poly}"/>')
            for x, y in points:
                lines.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.5" fill="{color}"/>')
            lx, ly = points[0]
            lines.append(f'<text x="{lx:.1f}" y="{ly-8:.1f}" fill="{color}">{estimator}</text>')

    lines.append(
        f'<text x="20" y="{margin_top + plot_height/2:.1f}" text-anchor="middle" transform="rotate(-90 20,{margin_top + plot_height/2:.1f})">{metric}</text>'
    )
    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")


def run_sweep(args: argparse.Namespace) -> list[dict]:
    rng = random.Random(args.seed)
    sources = make_sources()
    block_lengths = [int(x) for x in args.block_lengths.split(",")]
    per_rep_rows: list[dict] = []

    for rep in range(args.n):
        family_refs: dict[str, list[float]] = defaultdict(list)
        family_bits: dict[str, list[tuple]] = defaultdict(list)
        for source in sources:
            bits, ref = source.generator(args.length, rng)
            family_refs[source.family].append(ref)
            family_bits[source.family].append((source.name, bits, ref))

        for family in ("stationary", "nonstationary"):
            family_reference = mean(family_refs[family])
            pooled_bits = []
            for _, bits, _ in family_bits[family]:
                pooled_bits.extend(bits)

            ctw_est = ctw_entropy(pooled_bits, depth=args.ctw_depth)
            per_rep_rows.append(
                {
                    "rep": rep,
                    "family": family,
                    "estimator": "CTW",
                    "block_length": "full",
                    "reference_entropy": family_reference,
                    "estimate": ctw_est,
                    "error": ctw_est - family_reference,
                }
            )

            for block in block_lengths:
                if block >= args.length:
                    continue
                actw_fixed = fixed_block_actw_entropy(
                    pooled_bits, block_length=block, depth=max(1, args.ctw_depth - 1)
                )
                per_rep_rows.append(
                    {
                        "rep": rep,
                        "family": family,
                        "estimator": "ACTW_fixed",
                        "block_length": block,
                        "reference_entropy": family_reference,
                        "estimate": actw_fixed,
                        "error": actw_fixed - family_reference,
                    }
                )
                actw_random = random_window_actw_entropy(
                    pooled_bits,
                    min_window=max(256, block // 2),
                    max_window=min(args.length, block * 2),
                    samples=args.actw_samples,
                    depth=max(1, args.ctw_depth - 1),
                    rng=rng,
                )
                per_rep_rows.append(
                    {
                        "rep": rep,
                        "family": family,
                        "estimator": "ACTW_random",
                        "block_length": block,
                        "reference_entropy": family_reference,
                        "estimate": actw_random,
                        "error": actw_random - family_reference,
                    }
                )
                lz_est = lz_gkb_entropy(pooled_bits, window=block, max_match=args.lz_max_match)
                per_rep_rows.append(
                    {
                        "rep": rep,
                        "family": family,
                        "estimator": "LZ_GKB",
                        "block_length": block,
                        "reference_entropy": family_reference,
                        "estimate": lz_est,
                        "error": lz_est - family_reference,
                    }
                )

    summary: list[dict] = []
    grouped: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in per_rep_rows:
        if math.isnan(float(row["estimate"])):
            continue
        grouped[(row["family"], row["estimator"], str(row["block_length"]))].append(row)

    for (family, estimator, block_length), rows in sorted(grouped.items()):
        estimates = [float(r["estimate"]) for r in rows]
        errors = [float(r["error"]) for r in rows]
        summary.append(
            {
                "family": family,
                "estimator": estimator,
                "block_length": block_length,
                "mean_reference_entropy": mean(float(r["reference_entropy"]) for r in rows),
                "mean_estimate": mean(estimates),
                "bias": mean(errors),
                "std": pstdev(estimates),
                "var": pstdev(estimates) ** 2,
                "n_reps": len(rows),
            }
        )

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "block_length_raw.csv", per_rep_rows)
    write_csv(out / "block_length_summary.csv", summary)
    plot_rows = [r for r in summary if r["block_length"] != "full"]
    write_sensitivity_svg(out / "block_length_bias.svg", plot_rows, "bias", "Bias vs block length")
    write_sensitivity_svg(out / "block_length_mean.svg", plot_rows, "mean_estimate", "Mean estimate vs block length")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--length", type=int, default=15000)
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--ctw-depth", type=int, default=6)
    parser.add_argument("--lz-max-match", type=int, default=256)
    parser.add_argument("--actw-samples", type=int, default=32)
    parser.add_argument(
        "--block-lengths",
        default="512,1024,2048,4096,8192",
        help="comma-separated block/window lengths to sweep",
    )
    parser.add_argument("--output-dir", default="results")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_sweep(args)
    print("family,estimator,block_length,mean,bias,std,var,n_reps")
    for row in summary:
        print(
            f"{row['family']},{row['estimator']},{row['block_length']},"
            f"{row['mean_estimate']:.4f},{row['bias']:+.4f},"
            f"{row['std']:.4f},{row['var']:.5f},{row['n_reps']}"
        )
    print(f"\nWrote outputs to {args.output_dir}/")


if __name__ == "__main__":
    main()
