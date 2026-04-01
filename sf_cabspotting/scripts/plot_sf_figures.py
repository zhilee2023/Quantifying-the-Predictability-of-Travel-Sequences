"""
Figures for the San Francisco (Cabspotting) predictability experiment.

Outputs (under --workspace by default):
  - fig_bay_area_k_predictability_bar.{pdf,png}
  - fig_predictability_variation_geo.{pdf,png}  (optional, needs sf_dataset.csv)

Requires: matplotlib, numpy, pandas, geopandas; results under sfexp_result/ (HPC copy).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
from bay_area_map_style import (
    AREA_EDGECOLOR,
    AREA_LINEWIDTH,
    basemap_matching_partition_figure,
)


def plot_k_bar(*, workspace: Path, run_dir: Path) -> None:
    sys.path.insert(0, str(workspace / "src"))
    from make_sf_paper_assets import weighted_group_average
    from make_sf_summary_figure import aggregate_user_results

    plt.rcParams.update(
        {
            "font.size": 13,
            "axes.labelsize": 15,
            "xtick.labelsize": 12,
            "ytick.labelsize": 13,
            "legend.fontsize": 12,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    _, summary_df = aggregate_user_results(run_dir)
    if summary_df.empty:
        raise SystemExit(f"No user_level_results under {run_dir}")
    vq = summary_df.loc[summary_df["method"] == "vq_ctw"].copy()
    avg = weighted_group_average(
        vq,
        group_cols=["sample_interval_min", "tolerance_km", "codebook_size"],
        value_cols=["predictability"],
    ).sort_values(["sample_interval_min", "tolerance_km", "codebook_size"])
    pivot = avg.pivot(
        index=["sample_interval_min", "tolerance_km"],
        columns="codebook_size",
        values="predictability",
    ) * 100
    preferred = [256.0, 512.0, 1024.0]
    present = [c for c in preferred if c in pivot.columns]
    if not present:
        present = sorted(float(c) for c in pivot.columns)
    pivot = pivot[present]
    labels = [f"{int(i)} min\n{t:g} km" for i, t in pivot.index]
    x = np.arange(len(labels))
    n_k = len(pivot.columns)
    width = min(0.28, 0.75 / max(n_k, 1))
    colors = ["#4E79A7", "#F28E2B", "#59A14F"]
    fig, ax = plt.subplots(figsize=(10.5, 5.8), dpi=200)
    offsets = np.linspace(-(n_k - 1) / 2, (n_k - 1) / 2, n_k) if n_k else []
    for idx, k in enumerate(pivot.columns):
        color = colors[idx % len(colors)]
        ax.bar(
            x + offsets[idx] * width,
            pivot[k].to_numpy(),
            width=width,
            color=color,
            edgecolor="white",
            linewidth=0.8,
            label=f"K = {int(k)}",
            zorder=3,
        )
    ax.set_xlabel("Spatiotemporal resolution pair")
    ax.set_ylabel("Predictability (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.yaxis.grid(True, linestyle="--", linewidth=0.8, alpha=0.35, zorder=0)
    ax.xaxis.grid(False)
    ax.margins(x=0.03)
    ax.set_ylim(0, 100)
    ax.legend(
        ncol=min(3, n_k),
        loc="upper center",
        bbox_to_anchor=(0.5, 1.12),
        frameon=False,
        handlelength=1.4,
        columnspacing=1.6,
    )
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    stem = workspace / "fig_bay_area_k_predictability_bar"
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=260, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {stem}.pdf / .png")


def plot_geo_density(*, workspace: Path, data_csv: Path, scenario_dir: Path, checkpoint: Path) -> None:
    area, (x_min, x_max, y_min, y_max), _ = basemap_matching_partition_figure(scenario_dir, checkpoint)
    df = pd.read_csv(data_csv)
    df["sx"] = df["start_point"].str.extract(r"POINT\(([-0-9.]+) ([-0-9.]+)\)")[0].astype(float)
    df["sy"] = df["start_point"].str.extract(r"POINT\(([-0-9.]+) ([-0-9.]+)\)")[1].astype(float)
    df["ex"] = df["end_point"].str.extract(r"POINT\(([-0-9.]+) ([-0-9.]+)\)")[0].astype(float)
    df["ey"] = df["end_point"].str.extract(r"POINT\(([-0-9.]+) ([-0-9.]+)\)")[1].astype(float)
    pts = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy(
            np.r_[df["sx"].to_numpy(), df["ex"].to_numpy()],
            np.r_[df["sy"].to_numpy(), df["ey"].to_numpy()],
        ),
        crs="EPSG:4326",
    )
    inside = gpd.sjoin(pts, area[["geometry"]], how="inner", predicate="within")
    lon = inside.geometry.x.to_numpy()
    lat = inside.geometry.y.to_numpy()
    viridis_bg = plt.cm.viridis(0.0)
    x_span = x_max - x_min
    y_span = y_max - y_min
    target_cell_deg = min(x_span, y_span) / 240
    nx = max(1, int(np.ceil(x_span / target_cell_deg)))
    ny = max(1, int(np.ceil(y_span / target_cell_deg)))
    h, xe, ye = np.histogram2d(lon, lat, bins=[nx, ny], range=[[x_min, x_max], [y_min, y_max]])
    h_plot = np.ma.masked_where(h == 0, h)
    vmax = float(h.max()) if h.size else 1.0
    h_for_mesh = h_plot.T

    fig, ax = plt.subplots(1, 1, figsize=(7.0, 8.0), dpi=150)
    fig.subplots_adjust(left=0.04, right=0.82, top=0.96, bottom=0.04)
    area.plot(ax=ax, facecolor=viridis_bg, edgecolor="none", zorder=1)
    im = ax.pcolormesh(
        xe,
        ye,
        h_for_mesh,
        cmap="viridis",
        norm=LogNorm(vmin=1.0, vmax=max(vmax, 1.0)),
        shading="flat",
        alpha=0.78,
        zorder=2,
        rasterized=True,
    )
    area.boundary.plot(ax=ax, edgecolor=AREA_EDGECOLOR, linewidth=AREA_LINEWIDTH, zorder=3)
    _cbw = 0.028
    _cbh = 0.2
    cax = fig.add_axes([0.84, 0.62, _cbw, _cbh])
    cb = fig.colorbar(im, cax=cax)
    cb.ax.tick_params(labelsize=7)
    cb.ax.set_title("count", fontsize=9, pad=4)
    ax.set_axis_off()
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    stem = workspace / "fig_predictability_variation_geo"
    fig.savefig(stem.with_suffix(".pdf"), dpi=500, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=260, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {stem}.pdf / .png")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SF experiment figures (single entry point).")
    here = Path(__file__).resolve().parents[1]
    p.add_argument("--workspace", type=Path, default=here, help="sf_cabspotting folder root (contains src/, scripts/).")
    p.add_argument("--run-dir", type=Path, default=None, help="Folder with **/user_level_results.csv (default: workspace/sfexp_result).")
    p.add_argument(
        "--mode",
        choices=["bar", "geo", "all"],
        default="all",
        help="bar: K predictability bars; geo: heatmap + basemap; all: both.",
    )
    p.add_argument("--data-csv", type=Path, default=None, help="sf_dataset.csv for geo mode.")
    p.add_argument("--scenario-dir", type=Path, default=None, help="One tol_* scenario with latent_code_occurrences.csv.")
    p.add_argument("--checkpoint", type=Path, default=None, help="Matching .pt for scenario meta.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ws = args.workspace.resolve()
    run_dir = args.run_dir or (ws / "sfexp_result")
    if args.mode in ("bar", "all"):
        plot_k_bar(workspace=ws, run_dir=run_dir)
    if args.mode in ("geo", "all"):
        data_csv = args.data_csv or (ws / "data" / "sf_dataset.csv")
        sc = args.scenario_dir or (run_dir / "sf_run_1980198_task_0" / "tol_2p5km_int_5min")
        ck = args.checkpoint or (sc / "vqvae_tol2p5km_int5min_linear_K256.pt")
        if not data_csv.is_file():
            print(f"[skip geo] missing {data_csv}", file=sys.stderr)
            return
        if not ck.is_file():
            print(f"[skip geo] missing checkpoint {ck}", file=sys.stderr)
            return
        plot_geo_density(workspace=ws, data_csv=data_csv, scenario_dir=sc, checkpoint=ck)


if __name__ == "__main__":
    main()
