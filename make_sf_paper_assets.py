from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from make_sf_summary_figure import METHOD_COLORS, METHOD_LABELS, aggregate_user_results

SF_BOUNDARY_GEOJSON_URL = (
    "https://services1-nocdn.arcgis.com/0MSEUqKaxRlEPj5g/ArcGIS/rest/services/"
    "San_Francisco_WFL1/FeatureServer/6/query?where=1%3D1&outFields=*&returnGeometry=true&f=geojson"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate SF paper figures and tables.")
    parser.add_argument("run_dir", type=Path, help="Root directory of the SF experiment outputs.")
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("c:/Users/zhile/Desktop/rl_c/sf_dataset.csv"),
        help="Path to the SF dataset CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for generated paper assets. Defaults to run_dir.",
    )
    return parser.parse_args()


def weighted_group_average(
    df: pd.DataFrame,
    group_cols: list[str],
    value_cols: list[str],
    weight_col: str = "num_points",
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for keys, group in df.groupby(group_cols, dropna=False, sort=True):
        row = dict(zip(group_cols, keys))
        weights = group[weight_col].astype(float)
        row[weight_col] = float(weights.sum())
        for value_col in value_cols:
            values = group[value_col].astype(float)
            valid = ~(values.isna() | weights.isna())
            row[value_col] = float(np.average(values[valid], weights=weights[valid])) if valid.any() else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def build_dataset_overview(data_path: Path, output_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(data_path, usecols=["trajectory", "timestamp", "start_point", "end_point"])
    start_xy = df["start_point"].str.extract(r"POINT\(([-0-9.]+) ([-0-9.]+)\)").astype(float)
    end_xy = df["end_point"].str.extract(r"POINT\(([-0-9.]+) ([-0-9.]+)\)").astype(float)
    taxi_ids = df["trajectory"].astype(str).str.strip().str.rsplit("_", n=1).str[0]

    stats = pd.DataFrame(
        [
            {
                "num_records": int(len(df)),
                "num_taxis": int(taxi_ids.nunique()),
                "time_start": str(df["timestamp"].min()),
                "time_end": str(df["timestamp"].max()),
                "longitude_min": float(start_xy[0].min()),
                "longitude_max": float(start_xy[0].max()),
                "latitude_min": float(start_xy[1].min()),
                "latitude_max": float(start_xy[1].max()),
            }
        ]
    )
    stats.to_csv(output_dir / "sf_dataset_overview_stats.csv", index=False)

    fig, ax = plt.subplots(figsize=(8.5, 8))
    try:
        sf_boundary = gpd.read_file(SF_BOUNDARY_GEOJSON_URL).to_crs("EPSG:4326")
        sf_boundary.boundary.plot(ax=ax, color="#202020", linewidth=1.6, zorder=3)
        bounds = sf_boundary.total_bounds
        ax.set_xlim(float(bounds[0]) - 0.01, float(bounds[2]) + 0.01)
        ax.set_ylim(float(bounds[1]) - 0.01, float(bounds[3]) + 0.01)
    except Exception:
        ax.set_xlim(float(start_xy[0].min()) - 0.02, float(start_xy[0].max()) + 0.02)
        ax.set_ylim(float(start_xy[1].min()) - 0.02, float(start_xy[1].max()) + 0.02)

    ax.scatter(
        start_xy[0].to_numpy(),
        start_xy[1].to_numpy(),
        s=0.12,
        alpha=0.05,
        color="#1f77b4",
        linewidths=0,
        rasterized=True,
        zorder=1,
        label="Trip start GPS points",
    )
    ax.scatter(
        end_xy[0].to_numpy(),
        end_xy[1].to_numpy(),
        s=0.12,
        alpha=0.04,
        color="#d62728",
        linewidths=0,
        rasterized=True,
        zorder=2,
        label="Trip end GPS points",
    )
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Cabspotting / San Francisco boundary and GPS points")
    ax.grid(alpha=0.2)
    ax.legend(loc="lower left", frameon=True)
    info_text = "\n".join(
        [
            f"Records: {len(df):,}",
            f"Taxis: {taxi_ids.nunique():,}",
            f"Time span: {df['timestamp'].min()} to {df['timestamp'].max()}",
            "Boundary: official San Francisco ArcGIS layer",
            "Original format: start/end WKT points per record",
        ]
    )
    ax.text(
        0.02,
        0.98,
        info_text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "#b0b0b0"},
    )
    fig.tight_layout()
    fig.savefig(output_dir / "sf_dataset_overview_map.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    return stats


def build_best_vq_table(summary_df: pd.DataFrame) -> pd.DataFrame:
    vq_df = summary_df.loc[summary_df["method"] == "vq_ctw"].copy()
    vq_avg = weighted_group_average(
        vq_df,
        group_cols=["sample_interval_min", "tolerance_km", "codebook_size"],
        value_cols=["predictability", "reconstruction_rmse_km"],
    )
    best_vq = (
        vq_avg.sort_values(
            by=["sample_interval_min", "tolerance_km", "predictability", "codebook_size"],
            ascending=[True, True, False, True],
        )
        .drop_duplicates(subset=["sample_interval_min", "tolerance_km"], keep="first")
        .rename(
            columns={
                "codebook_size": "best_codebook_size",
                "predictability": "best_vq_predictability",
                "reconstruction_rmse_km": "best_vq_rmse_km",
            }
        )
        .sort_values(by=["sample_interval_min", "tolerance_km"])
        .reset_index(drop=True)
    )
    return best_vq


def build_vq_sensitivity_assets(summary_df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    vq_df = summary_df.loc[summary_df["method"] == "vq_ctw"].copy()
    best_vq = build_best_vq_table(summary_df)
    best_vq.to_csv(output_dir / "sf_vq_bestk_by_resolution_table.csv", index=False)

    vq_k_avg = weighted_group_average(
        vq_df,
        group_cols=["sample_interval_min", "codebook_size"],
        value_cols=["predictability"],
    ).sort_values(by=["sample_interval_min", "codebook_size"])

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    for tol_km, group in best_vq.groupby("tolerance_km", sort=True):
        ax.plot(
            group["sample_interval_min"],
            group["best_vq_predictability"],
            marker="o",
            linewidth=2,
            label=f"{tol_km:g} km",
        )
    ax.set_xlabel("Sampling interval (min)")
    ax.set_ylabel("Best VQ-CTW predictability")
    ax.set_title("(a) Predictability vs. sampling interval")
    ax.grid(True, alpha=0.3)
    ax.legend(title="Tolerance")

    ax = axes[1]
    for interval, group in vq_k_avg.groupby("sample_interval_min", sort=True):
        ax.plot(
            group["codebook_size"],
            group["predictability"],
            marker="s",
            linewidth=2,
            label=f"{int(interval)} min",
        )
    ax.set_xlabel("Codebook size K")
    ax.set_ylabel("Mean VQ-CTW predictability")
    ax.set_title("(b) Predictability vs. codebook size")
    ax.grid(True, alpha=0.3)
    ax.legend(title="Sampling interval")

    fig.tight_layout()
    fig.savefig(output_dir / "sf_vq_sensitivity.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    return best_vq


def build_method_comparison_assets(summary_df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    best_vq = build_best_vq_table(summary_df)
    non_vq = weighted_group_average(
        summary_df.loc[summary_df["method"] != "vq_ctw"].copy(),
        group_cols=["sample_interval_min", "tolerance_km", "method"],
        value_cols=["predictability"],
    )

    method_table = (
        non_vq.pivot_table(
            index=["sample_interval_min", "tolerance_km"],
            columns="method",
            values="predictability",
        )
        .reset_index()
        .rename_axis(None, axis=1)
        .merge(best_vq[["sample_interval_min", "tolerance_km", "best_codebook_size", "best_vq_predictability"]], on=["sample_interval_min", "tolerance_km"])
        .rename(
            columns={
                "direct_ctw": "direct_ctw_predictability",
                "markov_order_1": "markov_order_1_predictability",
                "markov_order_2": "markov_order_2_predictability",
            }
        )
        .sort_values(by=["sample_interval_min", "tolerance_km"])
        .reset_index(drop=True)
    )
    method_table.to_csv(output_dir / "sf_method_comparison_bestk_table.csv", index=False)

    plot_df = method_table.copy()
    plot_df["label"] = plot_df.apply(lambda r: f"{int(r['sample_interval_min'])}m\n{r['tolerance_km']:g}km", axis=1)
    x = np.arange(len(plot_df))
    width = 0.2

    fig, axes = plt.subplots(1, 2, figsize=(16, 5.5))

    ax = axes[0]
    ax.bar(x - 1.5 * width, plot_df["direct_ctw_predictability"], width, label="Direct CTW", color=METHOD_COLORS["direct_ctw"])
    ax.bar(x - 0.5 * width, plot_df["markov_order_1_predictability"], width, label="Markov-1", color=METHOD_COLORS["markov_order_1"])
    ax.bar(x + 0.5 * width, plot_df["markov_order_2_predictability"], width, label="Markov-2", color=METHOD_COLORS["markov_order_2"])
    ax.bar(x + 1.5 * width, plot_df["best_vq_predictability"], width, label="Best VQ-CTW", color=METHOD_COLORS["vq_ctw"])
    ax.set_xticks(x, plot_df["label"])
    ax.set_ylabel("Predictability")
    ax.set_title("(a) Method comparison with best K")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1]
    gap = plot_df["best_vq_predictability"] - plot_df["direct_ctw_predictability"]
    colors = np.where(gap >= 0, "#2ca02c", "#d62728")
    ax.bar(x, gap, color=colors)
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_xticks(x, plot_df["label"])
    ax.set_ylabel("Best VQ-CTW minus Direct CTW")
    ax.set_title("(b) Relative gain over direct discretization")
    ax.grid(True, axis="y", alpha=0.3)

    for ax in axes:
        ax.tick_params(axis="x", labelsize=8)

    fig.tight_layout()
    fig.savefig(output_dir / "sf_method_comparison_bestk.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    return method_table


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    output_dir = (args.output_dir or run_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    _, summary_df = aggregate_user_results(run_dir)
    if summary_df.empty:
        raise SystemExit(f"No user-level SF experiment results found under {run_dir}")

    build_dataset_overview(args.data_path.resolve(), output_dir)
    build_vq_sensitivity_assets(summary_df, output_dir)
    build_method_comparison_assets(summary_df, output_dir)


if __name__ == "__main__":
    main()
