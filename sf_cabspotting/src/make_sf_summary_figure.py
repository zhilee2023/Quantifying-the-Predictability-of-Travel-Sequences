from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METHOD_ORDER = ["direct_ctw", "markov_order_1", "markov_order_2", "vq_ctw"]
METHOD_LABELS = {
    "direct_ctw": "Direct CTW",
    "markov_order_1": "Markov-1",
    "markov_order_2": "Markov-2",
    "vq_ctw": "VQ-CTW",
}
METHOD_COLORS = {
    "direct_ctw": "#1f77b4",
    "markov_order_1": "#ff7f0e",
    "markov_order_2": "#2ca02c",
    "vq_ctw": "#d62728",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build one summary figure for all SF experiments in a run folder.")
    parser.add_argument("run_dir", type=Path, help="Run root under sf_results, e.g. sf_results/sf_smoke")
    return parser.parse_args()


def weighted_average(series: pd.Series, weights: pd.Series) -> float:
    valid = ~(series.isna() | weights.isna())
    if not valid.any():
        return float("nan")
    return float(np.average(series[valid], weights=weights[valid]))


def aggregate_user_results(run_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    for user_path in sorted(run_dir.glob("**/user_level_results.csv")):
        scenario_dir = user_path.parent
        chunk = pd.read_csv(user_path)
        if chunk.empty:
            continue
        chunk.insert(0, "scenario_subdir", scenario_dir.name)
        chunk.insert(0, "run_subdir", scenario_dir.parent.name)
        frames.append(chunk)

    if not frames:
        return pd.DataFrame(), pd.DataFrame()

    user_df = pd.concat(frames, ignore_index=True)
    rows: list[dict[str, object]] = []
    group_cols = [
        "run_subdir",
        "scenario_subdir",
        "sample_interval_min",
        "interpolation_method",
        "tolerance_km",
        "method",
        "codebook_size",
    ]
    for keys, group in user_df.groupby(group_cols, dropna=False, sort=True):
        row = dict(zip(group_cols, keys))
        row["num_points"] = float(group["num_points"].sum())
        row["predictability"] = weighted_average(group["predictability"], group["num_points"])
        row["entropy_rate_bits"] = weighted_average(group["entropy_rate_bits"], group["num_points"])
        row["reconstruction_rmse_km"] = weighted_average(group["reconstruction_rmse_km"], group["num_points"])
        row["actual_distortion_km"] = weighted_average(group["actual_distortion_km"], group["num_points"])
        row["num_users"] = int(group["user_id"].nunique())
        rows.append(row)

    summary_df = pd.DataFrame(rows)
    summary_df["scenario_label"] = summary_df.apply(
        lambda r: f"{r['sample_interval_min']:.0f} min\n{r['tolerance_km']:g} km", axis=1
    )
    summary_df["experiment_label"] = summary_df.apply(
        lambda r: f"{r['sample_interval_min']:.0f} min\n{r['tolerance_km']:g} km\n{r['interpolation_method']}",
        axis=1,
    )
    return user_df, summary_df


def scenario_sort_key(row: pd.Series) -> tuple[float, float, str]:
    return (float(row["sample_interval_min"]), float(row["tolerance_km"]), str(row["interpolation_method"]))


def build_figure(run_dir: Path, metadata: dict[str, object], user_df: pd.DataFrame, summary_df: pd.DataFrame) -> None:
    if summary_df.empty:
        return

    experiment_cols = [
        "run_subdir",
        "scenario_subdir",
        "sample_interval_min",
        "tolerance_km",
        "interpolation_method",
        "experiment_label",
    ]
    best_vq_df = (
        summary_df.loc[summary_df["method"] == "vq_ctw"]
        .sort_values(
            by=["sample_interval_min", "tolerance_km", "interpolation_method", "predictability", "codebook_size"],
            ascending=[True, True, True, False, True],
        )
        .drop_duplicates(subset=["run_subdir", "scenario_subdir", "interpolation_method"], keep="first")
    )
    non_vq_df = summary_df.loc[summary_df["method"] != "vq_ctw"].copy()
    plot_df = pd.concat([non_vq_df, best_vq_df], ignore_index=True)

    scenario_order_df = (
        plot_df[experiment_cols]
        .sort_values(by=["sample_interval_min", "tolerance_km", "interpolation_method"])
        .drop_duplicates(subset=["run_subdir", "scenario_subdir", "interpolation_method"], keep="first")
        .sort_values(by=["sample_interval_min", "tolerance_km", "interpolation_method"])
    )
    experiment_order = [
        f"{row.run_subdir}/{row.scenario_subdir}/{row.interpolation_method}"
        for row in scenario_order_df.itertuples(index=False)
    ]
    experiment_labels = scenario_order_df["experiment_label"].tolist()
    plot_df["experiment_key"] = (
        plot_df["run_subdir"].astype(str)
        + "/"
        + plot_df["scenario_subdir"].astype(str)
        + "/"
        + plot_df["interpolation_method"].astype(str)
    )
    x = np.arange(len(experiment_order))

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # Panel A: predictability across all scenarios
    ax = axes[0]
    for method in METHOD_ORDER:
        chunk = plot_df.loc[plot_df["method"] == method].copy()
        if chunk.empty:
            continue
        chunk["experiment_key"] = pd.Categorical(chunk["experiment_key"], categories=experiment_order, ordered=True)
        chunk = chunk.sort_values("experiment_key")
        ax.plot(
            x[: len(chunk)],
            chunk["predictability"],
            marker="o",
            linewidth=2,
            label=METHOD_LABELS[method],
            color=METHOD_COLORS[method],
        )
    ax.set_xticks(x, experiment_labels)
    ax.set_ylabel("Weighted mean predictability")
    ax.set_title("(a) Predictability across experiments")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    # Panel B: distortion/RMSE across all scenarios
    ax = axes[1]
    for method in METHOD_ORDER:
        chunk = plot_df.loc[plot_df["method"] == method].copy()
        if chunk.empty:
            continue
        chunk["experiment_key"] = pd.Categorical(chunk["experiment_key"], categories=experiment_order, ordered=True)
        chunk = chunk.sort_values("experiment_key")
        ax.plot(
            x[: len(chunk)],
            chunk["reconstruction_rmse_km"],
            marker="s",
            linewidth=2,
            label=METHOD_LABELS[method],
            color=METHOD_COLORS[method],
        )
    ax.set_xticks(x, experiment_labels)
    ax.set_ylabel("Weighted mean reconstruction RMSE (km)")
    ax.set_title("(b) Distortion across experiments")
    ax.grid(True, alpha=0.3)

    # Panel C: pooled user-level predictability distribution
    ax = axes[2]
    method_positions = np.arange(len(METHOD_ORDER))
    for idx, method in enumerate(METHOD_ORDER):
        chunk = user_df.loc[user_df["method"] == method].copy()
        if chunk.empty:
            continue
        values = chunk["predictability"].to_numpy(dtype=float)
        jitter = np.linspace(-0.12, 0.12, num=len(values)) if len(values) > 1 else np.array([0.0])
        ax.scatter(
            np.full(len(values), idx, dtype=float) + jitter,
            values,
            s=28,
            alpha=0.85,
            color=METHOD_COLORS[method],
            label=METHOD_LABELS[method] if idx == 0 else None,
        )
        ax.hlines(np.mean(values), idx - 0.22, idx + 0.22, colors="black", linewidth=2)
    ax.set_xticks(method_positions, [METHOD_LABELS[m] for m in METHOD_ORDER], rotation=15)
    ax.set_ylabel("User-level predictability")
    ax.set_title("(c) User-level spread across all experiments")
    ax.grid(True, alpha=0.3)

    dataset_label = str(metadata.get("dataset_label", metadata.get("dataset", "SF experiments")))
    fig.suptitle(f"{dataset_label}: summary of all experiments", fontsize=14, y=1.02)
    for ax in axes[:2]:
        ax.tick_params(axis="x", rotation=65, labelsize=8)
    fig.tight_layout()
    fig.savefig(run_dir / "sf_all_experiments_summary.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def build_direct_vs_vq_table(summary_df: pd.DataFrame) -> pd.DataFrame:
    if summary_df.empty:
        return pd.DataFrame()

    base_cols = ["run_subdir", "scenario_subdir", "sample_interval_min", "interpolation_method", "tolerance_km"]
    direct = (
        summary_df.loc[summary_df["method"] == "direct_ctw", base_cols + ["predictability", "reconstruction_rmse_km"]]
        .rename(
            columns={
                "predictability": "direct_predictability",
                "reconstruction_rmse_km": "direct_rmse_km",
            }
        )
    )
    best_vq = (
        summary_df.loc[summary_df["method"] == "vq_ctw"]
        .sort_values(by=base_cols + ["predictability", "codebook_size"], ascending=[True, True, True, True, True, False, True])
        .drop_duplicates(subset=base_cols, keep="first")
        .loc[:, base_cols + ["codebook_size", "predictability", "reconstruction_rmse_km"]]
        .rename(
            columns={
                "codebook_size": "best_vq_codebook_size",
                "predictability": "best_vq_predictability",
                "reconstruction_rmse_km": "best_vq_rmse_km",
            }
        )
    )
    comparison = direct.merge(best_vq, on=base_cols, how="outer")
    comparison["predictability_gap_direct_minus_best_vq"] = (
        comparison["direct_predictability"] - comparison["best_vq_predictability"]
    )
    return comparison.sort_values(by=["sample_interval_min", "tolerance_km", "interpolation_method"])


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    metadata_path = run_dir / "run_metadata.json"
    metadata = {}
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    user_df, summary_df = aggregate_user_results(run_dir)
    if summary_df.empty:
        raise SystemExit(f"No scenario user results found under {run_dir}")

    summary_df = summary_df.sort_values(
        by=["sample_interval_min", "tolerance_km", "interpolation_method", "method", "codebook_size"],
        na_position="last",
    )
    summary_df.to_csv(run_dir / "sf_all_experiments_summary_table.csv", index=False)
    build_direct_vs_vq_table(summary_df).to_csv(run_dir / "sf_direct_vs_best_vq_table.csv", index=False)
    build_figure(run_dir, metadata, user_df, summary_df)


if __name__ == "__main__":
    main()
