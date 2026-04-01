"""
Grouped bar chart of VQ-CTW predictability vs. K (same style as plot.ipynb cell 1),
with a Bay Area county outline as an inset basemap.

Uses aggregate_user_results(run_dir) like the notebook; default run_dir is sf_results_hpc_2.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
QP = Path(__file__).resolve().parent
sys.path.insert(0, str(QP))

from make_sf_paper_assets import weighted_group_average
from make_sf_summary_figure import aggregate_user_results

BAY_AREA_COUNTIES = [
    "Alameda",
    "Contra Costa",
    "Marin",
    "Napa",
    "San Francisco",
    "San Mateo",
    "Santa Clara",
    "Solano",
    "Sonoma",
]


def _county_name_col(gdf: gpd.GeoDataFrame) -> Optional[str]:
    for candidate in ["NAME", "County_Name", "county_name", "name", "CDT_NAME_SHORT"]:
        if candidate in gdf.columns:
            return candidate
    return None


def load_bay_area_outline(workspace_root: Path) -> gpd.GeoDataFrame:
    cache = workspace_root / "bay_area_counties_ca.geojson"
    if not cache.exists():
        raise FileNotFoundError(f"Missing {cache}; expected cached Bay Area counties GeoJSON.")
    counties = gpd.read_file(cache)
    name_col = _county_name_col(counties)
    if name_col is None:
        raise KeyError("Could not locate county name column in GeoJSON.")
    bay = counties.loc[counties[name_col].isin(BAY_AREA_COUNTIES)].copy()
    if bay.empty:
        raise ValueError("Bay Area county filter returned no polygons.")
    return bay.to_crs("EPSG:4326")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Bar chart with Bay Area inset (plot.ipynb cell 1 style).")
    p.add_argument(
        "--run-dir",
        type=Path,
        default=WORKSPACE_ROOT / "sf_results_hpc_2",
        help="Experiment root containing **/user_level_results.csv (e.g. sf_results_hpc_2).",
    )
    p.add_argument(
        "--output-stem",
        type=Path,
        default=WORKSPACE_ROOT / "fig_bay_area_k_predictability_bar",
        help="Output path without extension (writes .pdf and .png).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    out = args.output_stem.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

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

    pivot = pivot[[256.0, 512.0, 1024.0]]
    labels = [f"{int(i)} min\n{t:g} km" for i, t in pivot.index]
    x = np.arange(len(labels))
    width = 0.23
    colors = ["#4E79A7", "#F28E2B", "#59A14F"]

    fig, ax = plt.subplots(figsize=(10.5, 5.8), dpi=200)

    for idx, (k, color) in enumerate(zip(pivot.columns, colors)):
        ax.bar(
            x + (idx - 1) * width,
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
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.12),
        frameon=False,
        handlelength=1.4,
        columnspacing=1.6,
    )

    bay = load_bay_area_outline(WORKSPACE_ROOT)
    ax_inset = inset_axes(ax, width="30%", height="36%", loc="lower right", borderpad=1.2)
    bay.plot(
        ax=ax_inset,
        facecolor="#E3ECF4",
        edgecolor="#4E79A7",
        linewidth=0.55,
        alpha=0.92,
        zorder=1,
    )
    ax_inset.set_axis_off()
    bounds = bay.total_bounds
    ax_inset.set_xlim(bounds[0], bounds[2])
    ax_inset.set_ylim(bounds[1], bounds[3])
    ax_inset.set_aspect("equal", adjustable="box")
    ax_inset.text(
        0.5,
        0.02,
        "Bay Area",
        transform=ax_inset.transAxes,
        ha="center",
        va="bottom",
        fontsize=9,
        color="#2d3d4d",
    )

    fig.subplots_adjust(left=0.09, right=0.98, top=0.88, bottom=0.11)
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=260, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out.with_suffix('.pdf')} and {out.with_suffix('.png')}")


if __name__ == "__main__":
    main()
