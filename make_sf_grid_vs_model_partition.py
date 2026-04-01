from __future__ import annotations

"""
Figure: direct spatial grid vs. VQ-based partition (learned representation on the map)

What “learned representation” means here
    During EC–VQ-VAE training, each trajectory point is encoded and quantized to a discrete
    codebook index (VQ code / raw_code). That index is the learned representation used in
    downstream predictability analysis. The figure does **not** re-run the encoder at plot
    time; it reads **precomputed** assignments from the experiment outputs.

Data source (per checkpoint scenario directory)
    ``latent_code_occurrences.csv`` lists, for each retained sample point: planar coordinates
    ``x_km``, ``y_km`` (UTM zone 10, kilometers), lon/lat, and ``raw_code`` (the assigned
    VQ index). Rows are filtered to match the checkpoint’s ``*_meta.json`` experiment settings
    (sample interval, interpolation, tolerance_km, codebook size).

Panel (a) — direct spatial partition
    Uses the **same** tolerance as the experiment: ``tolerance_km`` from metadata. Points are
    binned with ``cell_x = floor(x_km / tolerance_km)``, ``cell_y = floor(y_km / tolerance_km)``.
    Occupied cells are drawn as polygons in display CRS; fill is **uniform** (partition
    geometry only, not visit-count coloring).

Panel (b) — VQ partition on a visualization grid
    Independently of the model’s training grid, a **finer** square grid is defined only for
    plotting (``--vq-display-cell-m`` in meters). For each display cell, all sample points
    falling in that cell are grouped; the **dominant** ``raw_code`` (plurality) labels the
    cell. Color maps **code index** ``0 … K-1`` to distinct colors (turbo, ``K`` =
    codebook size from checkpoint). **Purity** (share of points matching the dominant code)
    modulates opacity. Cells with no dominant-code mismatch outside ``[0, K)`` are rare;
    they are shown in gray if they occur.

Map and CRS
    Spatial operations use UTM EPSG:32610 (meters internally for cell geometry); optional
    display in WGS84 lon/lat (``--display-crs wgs84``). The map outline is Bay Area counties
    (optionally restricted to counties that intersect a sample of occurrence points).

Outputs
    PNG (and optional PDF). Summary text lists predictability metrics and cell counts from
    the same run metadata.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from urllib.request import urlretrieve

import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import numpy as np
import pandas as pd
from shapely.affinity import scale as geom_scale
from shapely.geometry import Point, box
import torch

from model import EC_VQVAE


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = (
    WORKSPACE_ROOT
    / "sf_results_hpc_2"
    / "sf_run_1980198_task_0"
    / "tol_2p5km_int_5min"
    / "vqvae_tol2p5km_int5min_linear_K256.pt"
)
CA_COUNTIES_GEOJSON_URL = "https://gis.data.ca.gov/api/download/v1/items/60b7e0f3d33b4064a4b43bf14589bfe3/geojson?layers=1"
BAY_AREA_COUNTIES_CACHE = WORKSPACE_ROOT / "bay_area_counties_ca.geojson"
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
UTM_CRS = "EPSG:32610"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a map-style direct-grid vs VQ-partition figure from one SF checkpoint.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help="Checkpoint .pt file for the desired SF scenario.",
    )
    parser.add_argument(
        "--output-stem",
        type=Path,
        default=WORKSPACE_ROOT / "fig_sf_2p5km_linear_K256_grid_vs_vq_partition",
        help="Output stem without extension.",
    )
    parser.add_argument(
        "--background-sample",
        type=int,
        default=180000,
        help="Maximum number of faint background points to draw.",
    )
    parser.add_argument(
        "--display-crs",
        choices=["wgs84", "utm_km"],
        default="wgs84",
        help="Display in WGS84 lon/lat or UTM kilometers. Grid construction always uses UTM kilometers.",
    )
    parser.add_argument(
        "--vq-display-cell-m",
        type=float,
        default=100.0,
        help="Visualization-only cell size for the right VQ panel, in meters.",
    )
    parser.add_argument(
        "--save-pdf",
        action="store_true",
        help="Also export a PDF version. Leave off for very fine grids to avoid slow vector output.",
    )
    return parser.parse_args()


def km_cell_columns(df: pd.DataFrame, x_col: str, y_col: str, cell_km: float) -> pd.DataFrame:
    out = df.copy()
    out["cell_x"] = np.floor(out[x_col] / cell_km).astype(int)
    out["cell_y"] = np.floor(out[y_col] / cell_km).astype(int)
    return out


def summarize_direct_cells(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby(["cell_x", "cell_y"], sort=True)
        .size()
        .rename("count")
        .reset_index()
        .sort_values("count", ascending=False)
        .reset_index(drop=True)
    )
    total = float(summary["count"].sum())
    summary["share"] = summary["count"] / total if total > 0 else 0.0
    return summary


def summarize_vq_cells(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Union[float, int]]] = []
    for (cell_x, cell_y), group in df.groupby(["cell_x", "cell_y"], sort=True):
        counts = group["raw_code"].value_counts()
        dominant_code = int(counts.index[0])
        dominant_count = int(counts.iloc[0])
        total_count = int(counts.sum())
        rows.append(
            {
                "cell_x": int(cell_x),
                "cell_y": int(cell_y),
                "count": total_count,
                "dominant_code": dominant_code,
                "purity": dominant_count / total_count,
            }
        )
    out = pd.DataFrame(rows).sort_values(["cell_y", "cell_x"]).reset_index(drop=True)
    return out


def checkpoint_meta_path(checkpoint: Path) -> Path:
    return checkpoint.with_name(checkpoint.stem + "_meta.json")


def load_checkpoint_and_meta(checkpoint: Path) -> Tuple[EC_VQVAE, dict]:
    meta_path = checkpoint_meta_path(checkpoint)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    model = EC_VQVAE(**meta["constructor_kwargs"])
    state = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model, meta


def load_occurrences(scenario_dir: Path, experiment: dict, codebook_size: int) -> pd.DataFrame:
    occ_path = scenario_dir / "latent_code_occurrences.csv"
    usecols = [
        "latitude",
        "longitude",
        "x_km",
        "y_km",
        "raw_code",
        "codebook_size",
        "sample_interval_min",
        "interpolation_method",
        "tolerance_km",
    ]
    occ_df = pd.read_csv(occ_path, usecols=usecols)
    sub = occ_df.loc[
        (occ_df["codebook_size"] == codebook_size)
        & (occ_df["sample_interval_min"] == int(experiment["sample_interval_min"]))
        & (occ_df["interpolation_method"] == str(experiment["interpolation_method"]))
        & np.isclose(occ_df["tolerance_km"].astype(float), float(experiment["tolerance_km"]))
    ].copy()
    if sub.empty:
        raise SystemExit(f"No filtered occurrences found in {occ_path}")
    return sub


def load_predictability_summary(scenario_dir: Path, experiment: dict, codebook_size: int) -> Tuple[float, float]:
    df = pd.read_csv(scenario_dir / "user_level_results.csv")
    mask_common = (
        (df["sample_interval_min"] == int(experiment["sample_interval_min"]))
        & (df["interpolation_method"] == str(experiment["interpolation_method"]))
        & np.isclose(df["tolerance_km"].astype(float), float(experiment["tolerance_km"]))
    )
    direct = df.loc[mask_common & (df["method"] == "direct_ctw")].copy()
    vq = df.loc[mask_common & (df["method"] == "vq_ctw") & (df["codebook_size"] == codebook_size)].copy()
    direct_pred = float(np.average(direct["predictability"], weights=direct["num_points"]))
    vq_pred = float(np.average(vq["predictability"], weights=vq["num_points"]))
    return direct_pred, vq_pred


def load_latent_summary(scenario_dir: Path, experiment: dict, codebook_size: int) -> pd.DataFrame:
    df = pd.read_csv(scenario_dir / "latent_code_summary.csv")
    return df.loc[
        (df["sample_interval_min"] == int(experiment["sample_interval_min"]))
        & (df["interpolation_method"] == str(experiment["interpolation_method"]))
        & (df["codebook_size"] == codebook_size)
    ].copy()


def sample_background_points(df: pd.DataFrame, max_points: int) -> np.ndarray:
    if len(df) <= max_points:
        sampled = df
    else:
        sampled = df.sample(n=max_points, random_state=42)
    return sampled


def county_name_col(gdf: gpd.GeoDataFrame) -> Optional[str]:
    for candidate in ["NAME", "County_Name", "county_name", "name", "CDT_NAME_SHORT"]:
        if candidate in gdf.columns:
            return candidate
    return None


def load_bay_area_wgs84() -> gpd.GeoDataFrame:
    if not BAY_AREA_COUNTIES_CACHE.exists():
        urlretrieve(CA_COUNTIES_GEOJSON_URL, BAY_AREA_COUNTIES_CACHE)
    counties = gpd.read_file(BAY_AREA_COUNTIES_CACHE)
    name_col = county_name_col(counties)
    if name_col is None:
        raise KeyError("Could not locate county name column in California county GeoJSON.")
    bay = counties.loc[counties[name_col].isin(BAY_AREA_COUNTIES)].copy()
    if bay.empty:
        raise ValueError("Bay Area county filter returned no polygons.")
    return bay.to_crs("EPSG:4326")


def load_sf_area_m() -> gpd.GeoDataFrame:
    return load_bay_area_wgs84().to_crs(UTM_CRS)


def load_sf_area_km() -> gpd.GeoDataFrame:
    area = load_sf_area_m().copy()
    area["geometry"] = area.geometry.apply(lambda geom: geom_scale(geom, xfact=1 / 1000, yfact=1 / 1000, origin=(0, 0)))
    return area


def filter_occurrences_inside_sf(df: pd.DataFrame) -> pd.DataFrame:
    points = gpd.GeoDataFrame(
        df.copy(),
        geometry=[Point(x * 1000.0, y * 1000.0) for x, y in zip(df["x_km"], df["y_km"])],
        crs=UTM_CRS,
    )
    area = load_sf_area_m()
    inside = gpd.sjoin(points, area[["geometry"]], how="inner", predicate="within")
    return pd.DataFrame(inside.drop(columns=["geometry", "index_right"]))


def involved_counties(occ_df: pd.DataFrame, sample_size: int = 120000) -> List[str]:
    counties = load_bay_area_wgs84()
    name_col = county_name_col(counties)
    if len(occ_df) > sample_size:
        occ_df = occ_df.sample(n=sample_size, random_state=42)
    points = gpd.GeoDataFrame(
        occ_df[["longitude", "latitude"]].copy(),
        geometry=gpd.points_from_xy(occ_df["longitude"], occ_df["latitude"]),
        crs="EPSG:4326",
    )
    joined = gpd.sjoin(points, counties[[name_col, "geometry"]], how="inner", predicate="within")
    return sorted(joined[name_col].dropna().astype(str).unique().tolist())


def load_display_area(display_crs: str, county_names: Optional[List[str]] = None) -> gpd.GeoDataFrame:
    if display_crs == "wgs84":
        area = load_bay_area_wgs84()
    else:
        area = load_sf_area_km()
    if county_names:
        name_col = county_name_col(area)
        area = area.loc[area[name_col].isin(county_names)].copy()
    return area


def build_display_background(sampled_df: pd.DataFrame, display_crs: str) -> np.ndarray:
    if display_crs == "wgs84":
        return sampled_df[["longitude", "latitude"]].to_numpy(dtype=float)
    return sampled_df[["x_km", "y_km"]].to_numpy(dtype=float)


def build_cell_geodf(cells_df: pd.DataFrame, cell_km: float, display_crs: str) -> gpd.GeoDataFrame:
    polygons = []
    for row in cells_df.itertuples(index=False):
        x0 = float(row.cell_x) * cell_km * 1000.0
        y0 = float(row.cell_y) * cell_km * 1000.0
        polygons.append(box(x0, y0, x0 + cell_km * 1000.0, y0 + cell_km * 1000.0))
    gdf = gpd.GeoDataFrame(cells_df.copy(), geometry=polygons, crs=UTM_CRS)
    if display_crs == "wgs84":
        return gdf.to_crs("EPSG:4326")
    gdf["geometry"] = gdf.geometry.apply(lambda geom: geom_scale(geom, xfact=1 / 1000, yfact=1 / 1000, origin=(0, 0)))
    return gdf


def build_cell_center_geodf(cells_df: pd.DataFrame, cell_km: float, display_crs: str) -> gpd.GeoDataFrame:
    centers_x = (cells_df["cell_x"].astype(float) + 0.5) * cell_km * 1000.0
    centers_y = (cells_df["cell_y"].astype(float) + 0.5) * cell_km * 1000.0
    gdf = gpd.GeoDataFrame(
        cells_df.copy(),
        geometry=gpd.points_from_xy(centers_x, centers_y),
        crs=UTM_CRS,
    )
    if display_crs == "wgs84":
        return gdf.to_crs("EPSG:4326")
    gdf["geometry"] = gdf.geometry.apply(lambda geom: geom_scale(geom, xfact=1 / 1000, yfact=1 / 1000, origin=(0, 0)))
    return gdf


def add_area_and_points(ax: plt.Axes, area: gpd.GeoDataFrame, background_xy: np.ndarray, show_points: bool) -> None:
    viridis_bg = plt.cm.viridis(0.0)
    moon_earth_gold = "#FFFACD"
    area.plot(ax=ax, facecolor=viridis_bg, edgecolor="gray", linewidth=0.55, alpha=1.0, zorder=1)
    if show_points and len(background_xy):
        ax.scatter(
            background_xy[:, 0],
            background_xy[:, 1],
            s=0.12,
            alpha=0.18,
            color=moon_earth_gold,
            linewidths=0,
            rasterized=True,
            zorder=2,
        )
    ax.set_axis_off()
    ax.set_aspect("equal", adjustable="box")


def full_codebook_color_map(codebook_size: int) -> dict[int, tuple]:
    """One distinct color per code index 0 .. codebook_size-1 (turbo, evenly spaced)."""
    n = max(int(codebook_size), 1)
    xs = np.linspace(0.0, 1.0, n) if n > 1 else np.array([0.5])
    arr = plt.cm.turbo(xs)
    return {i: tuple(arr[i]) for i in range(n)}


def add_direct_grid_panel(
    ax: plt.Axes,
    area: gpd.GeoDataFrame,
    background_xy: np.ndarray,
    direct_cells_gdf: gpd.GeoDataFrame,
) -> None:
    add_area_and_points(ax, area, background_xy, show_points=False)
    # Uniform fill: spatial partition only (no visit-count / density coloring).
    direct_cells_gdf.plot(
        ax=ax,
        color="#96b5d8",
        edgecolor=(1, 1, 1, 0.55),
        linewidth=0.25,
        alpha=0.72,
        zorder=3,
    )


def add_vq_partition_panel(
    ax: plt.Axes,
    area: gpd.GeoDataFrame,
    background_xy: np.ndarray,
    vq_cells_gdf: gpd.GeoDataFrame,
    codebook_size: int,
) -> None:
    add_area_and_points(ax, area, background_xy, show_points=False)
    color_map = full_codebook_color_map(codebook_size)
    out_of_range = "#9e9e9e"

    colors = []
    for row in vq_cells_gdf.itertuples(index=False):
        c = int(row.dominant_code)
        base = color_map[c] if 0 <= c < codebook_size else out_of_range
        alpha = 0.25 + 0.75 * float(row.purity)
        rgba = list(mcolors.to_rgba(base))
        rgba[3] = alpha
        colors.append(tuple(rgba))
    vq_cells_gdf.plot(
        ax=ax,
        color=colors,
        edgecolor=(1, 1, 1, 0.45),
        linewidth=0.22,
        zorder=3,
    )


def add_vq_partition_centers_panel(
    ax: plt.Axes,
    area: gpd.GeoDataFrame,
    background_xy: np.ndarray,
    vq_centers_gdf: gpd.GeoDataFrame,
    codebook_size: int,
    marker_size: float = 8.0,
) -> None:
    add_area_and_points(ax, area, background_xy, show_points=False)
    color_map = full_codebook_color_map(codebook_size)
    out_of_range = "#9e9e9e"

    colors = []
    alphas = []
    for row in vq_centers_gdf.itertuples(index=False):
        c = int(row.dominant_code)
        colors.append(color_map[c] if 0 <= c < codebook_size else out_of_range)
        alphas.append(0.25 + 0.75 * float(row.purity))
    xy = np.column_stack([vq_centers_gdf.geometry.x.to_numpy(), vq_centers_gdf.geometry.y.to_numpy()])
    ax.scatter(
        xy[:, 0],
        xy[:, 1],
        c=colors,
        s=marker_size,
        marker="s",
        alpha=alphas,
        linewidths=0,
        zorder=3,
        rasterized=True,
    )


def axis_limits(df: pd.DataFrame, cell_km: float) -> tuple[float, float, float, float]:
    cell_x = np.floor(df["x_km"] / cell_km).astype(int)
    cell_y = np.floor(df["y_km"] / cell_km).astype(int)
    margin = 2.0 * cell_km
    x_min = float(cell_x.min() * cell_km) - margin
    y_min = float(cell_y.min() * cell_km) - margin
    x_max = float((cell_x.max() + 1) * cell_km) + margin
    y_max = float((cell_y.max() + 1) * cell_km) + margin
    return x_min, x_max, y_min, y_max


def area_axis_limits(area: gpd.GeoDataFrame, margin_km: float = 1.5) -> tuple[float, float, float, float]:
    bounds = area.total_bounds
    return (
        float(bounds[0]) - margin_km,
        float(bounds[2]) + margin_km,
        float(bounds[1]) - margin_km,
        float(bounds[3]) + margin_km,
    )


def wgs84_axis_limits(area: gpd.GeoDataFrame, margin_deg: float = 0.01) -> tuple[float, float, float, float]:
    bounds = area.total_bounds
    return (
        float(bounds[0]) - margin_deg,
        float(bounds[2]) + margin_deg,
        float(bounds[1]) - margin_deg,
        float(bounds[3]) + margin_deg,
    )


def main() -> None:
    args = parse_args()
    checkpoint = args.checkpoint.resolve()
    scenario_dir = checkpoint.parent
    output_stem = args.output_stem.resolve()
    output_stem.parent.mkdir(parents=True, exist_ok=True)

    model, meta = load_checkpoint_and_meta(checkpoint)
    experiment = meta["experiment"]
    codebook_size = int(meta["constructor_kwargs"]["codebook_size"])
    occ_df = load_occurrences(scenario_dir, experiment, codebook_size)
    direct_pred, vq_pred = load_predictability_summary(scenario_dir, experiment, codebook_size)
    cell_km = float(experiment["tolerance_km"])
    vq_display_cell_km = float(args.vq_display_cell_m) / 1000.0
    occ_df = filter_occurrences_inside_sf(occ_df)
    direct_df = km_cell_columns(occ_df, "x_km", "y_km", cell_km)
    vq_df = km_cell_columns(occ_df, "x_km", "y_km", vq_display_cell_km)
    direct_cells = summarize_direct_cells(direct_df).reset_index()
    vq_cells = summarize_vq_cells(vq_df)
    active_codes = int(occ_df["raw_code"].nunique())
    mean_purity = float(vq_cells["purity"].mean())

    county_names = involved_counties(occ_df)
    area = load_display_area(args.display_crs, county_names=county_names)
    background_sample = sample_background_points(occ_df, args.background_sample)
    background_xy = build_display_background(background_sample, args.display_crs)
    direct_cells_gdf = build_cell_geodf(direct_cells, cell_km, args.display_crs)
    use_vq_centers = vq_display_cell_km <= 0.25
    if use_vq_centers:
        vq_cells_gdf = build_cell_center_geodf(vq_cells, vq_display_cell_km, args.display_crs)
    else:
        vq_cells_gdf = build_cell_geodf(vq_cells, vq_display_cell_km, args.display_crs)
    direct_cells_gdf = gpd.clip(direct_cells_gdf, area)
    vq_cells_gdf = gpd.clip(vq_cells_gdf, area)
    if args.display_crs == "wgs84":
        x_min, x_max, y_min, y_max = wgs84_axis_limits(area)
    else:
        x_min, x_max, y_min, y_max = area_axis_limits(area)

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(13.2, 7.2),
        dpi=170,
        sharex=True,
        sharey=True,
        gridspec_kw={"width_ratios": [1, 1], "wspace": 0.06},
    )
    add_direct_grid_panel(axes[0], area, background_xy, direct_cells_gdf)
    if use_vq_centers:
        marker_size = 8.0 if args.display_crs == "wgs84" else 4.0
        add_vq_partition_centers_panel(axes[1], area, background_xy, vq_cells_gdf, codebook_size, marker_size=marker_size)
    else:
        add_vq_partition_panel(axes[1], area, background_xy, vq_cells_gdf, codebook_size)

    axes[0].set_xlim(x_min, x_max)
    axes[0].set_ylim(y_min, y_max)

    fig.tight_layout(rect=[0, 0.06, 1.0, 0.98])
    captions = ("(a) Direct grid", "(b) VQ partition")
    y0 = min(ax.get_position().y0 for ax in axes)
    y_caption = y0 - 0.018
    caption_artists = []
    for ax, cap in zip(axes, captions):
        pos = ax.get_position()
        xc = pos.x0 + pos.width / 2
        t = fig.text(xc, y_caption, cap, ha="center", va="top", fontsize=12, transform=fig.transFigure)
        t.set_clip_on(False)
        caption_artists.append(t)
    fig.savefig(
        output_stem.with_suffix(".png"),
        dpi=260,
        bbox_inches="tight",
        bbox_extra_artists=caption_artists,
        pad_inches=0.12,
    )
    if args.save_pdf:
        fig.savefig(
            output_stem.with_suffix(".pdf"),
            dpi=260,
            bbox_inches="tight",
            bbox_extra_artists=caption_artists,
            pad_inches=0.12,
        )
    plt.close(fig)

    summary_lines = [
        f"checkpoint={checkpoint}",
        f"sample_interval_min={experiment['sample_interval_min']}",
        f"interpolation_method={experiment['interpolation_method']}",
        f"tolerance_km={experiment['tolerance_km']}",
        f"codebook_size={codebook_size}",
        f"direct_predictability={direct_pred:.6f}",
        f"vq_predictability={vq_pred:.6f}",
        f"direct_occupied_cells={len(direct_cells)}",
        f"vq_display_cell_m={args.vq_display_cell_m}",
        f"vq_display_cells={len(vq_cells)}",
        f"active_vq_codes={active_codes}",
        f"mean_cell_purity={mean_purity:.6f}",
    ]
    output_stem.with_name(output_stem.name + "_summary.txt").write_text("\n".join(summary_lines), encoding="utf-8")


if __name__ == "__main__":
    main()
