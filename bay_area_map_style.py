"""
Bay Area basemap helpers aligned with ``make_sf_grid_vs_model_partition.py`` (no torch).
Use the same county subset, axis limits, and draw styles as the grid-vs-VQ figures.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional
from urllib.request import urlretrieve

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.affinity import scale as geom_scale
from shapely.geometry import Point

UTM_CRS = "EPSG:32610"
CA_COUNTIES_GEOJSON_URL = (
    "https://gis.data.ca.gov/api/download/v1/items/60b7e0f3d33b4064a4b43bf14589bfe3/geojson?layers=1"
)
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

# Match add_area_and_points in make_sf_grid_vs_model_partition.py
AREA_EDGECOLOR = "gray"
AREA_LINEWIDTH = 0.55
SCATTER_S = 0.12
SCATTER_ALPHA = 0.18
MOON_EARTH_GOLD = "#FFFACD"
WGS84_MARGIN_DEG = 0.01


def workspace_root_from_here() -> Path:
    return Path(__file__).resolve().parents[1]


def county_name_col(gdf: gpd.GeoDataFrame) -> Optional[str]:
    for candidate in ["NAME", "County_Name", "county_name", "name", "CDT_NAME_SHORT"]:
        if candidate in gdf.columns:
            return candidate
    return None


def load_bay_area_wgs84(workspace_root: Path) -> gpd.GeoDataFrame:
    cache = workspace_root / "bay_area_counties_ca.geojson"
    if not cache.exists():
        urlretrieve(CA_COUNTIES_GEOJSON_URL, cache)
    counties = gpd.read_file(cache)
    name_col = county_name_col(counties)
    if name_col is None:
        raise KeyError("Could not locate county name column in California county GeoJSON.")
    bay = counties.loc[counties[name_col].isin(BAY_AREA_COUNTIES)].copy()
    if bay.empty:
        raise ValueError("Bay Area county filter returned no polygons.")
    return bay.to_crs("EPSG:4326")


def load_sf_area_m(workspace_root: Path) -> gpd.GeoDataFrame:
    return load_bay_area_wgs84(workspace_root).to_crs(UTM_CRS)


def filter_occurrences_inside_sf(df: pd.DataFrame, workspace_root: Path) -> pd.DataFrame:
    points = gpd.GeoDataFrame(
        df.copy(),
        geometry=[Point(x * 1000.0, y * 1000.0) for x, y in zip(df["x_km"], df["y_km"])],
        crs=UTM_CRS,
    )
    area = load_sf_area_m(workspace_root)
    inside = gpd.sjoin(points, area[["geometry"]], how="inner", predicate="within")
    return pd.DataFrame(inside.drop(columns=["geometry", "index_right"]))


def load_occurrences_filtered(scenario_dir: Path, checkpoint: Path) -> pd.DataFrame:
    """Same rows as ``load_occurrences`` + ``filter_occurrences_inside_sf`` in the figure script."""
    meta_path = checkpoint.with_name(checkpoint.stem + "_meta.json")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    experiment = meta["experiment"]
    codebook_size = int(meta["constructor_kwargs"]["codebook_size"])
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
        raise ValueError(f"No rows in {occ_path} matching checkpoint meta.")
    # scenario_dir = .../tol_* ; repo root is parents[2] (tol -> run -> sf_results_* -> rl_c)
    workspace_root = scenario_dir.resolve().parents[2]
    if not (workspace_root / "bay_area_counties_ca.geojson").exists():
        workspace_root = workspace_root_from_here()
    return filter_occurrences_inside_sf(sub, workspace_root)


def involved_counties(occ_df: pd.DataFrame, workspace_root: Path, sample_size: int = 120000) -> List[str]:
    counties = load_bay_area_wgs84(workspace_root)
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


def load_display_area_wgs84(workspace_root: Path, county_names: List[str]) -> gpd.GeoDataFrame:
    area = load_bay_area_wgs84(workspace_root)
    name_col = county_name_col(area)
    return area.loc[area[name_col].isin(county_names)].copy()


def wgs84_axis_limits(area: gpd.GeoDataFrame, margin_deg: float = WGS84_MARGIN_DEG) -> tuple[float, float, float, float]:
    bounds = area.total_bounds
    return (
        float(bounds[0]) - margin_deg,
        float(bounds[2]) + margin_deg,
        float(bounds[1]) - margin_deg,
        float(bounds[3]) + margin_deg,
    )


def basemap_matching_partition_figure(
    scenario_dir: Path,
    checkpoint: Path,
) -> tuple[gpd.GeoDataFrame, tuple[float, float, float, float], Path]:
    """
    Same ``area`` polygons and ``wgs84_axis_limits`` as ``make_sf_grid_vs_model_partition.py``
    for the given scenario + checkpoint (uses ``latent_code_occurrences.csv`` + 120k county sample).
    """
    workspace_root = scenario_dir.resolve().parents[2]
    if not (workspace_root / "bay_area_counties_ca.geojson").exists():
        workspace_root = workspace_root_from_here()
    occ_df = load_occurrences_filtered(scenario_dir, checkpoint)
    county_names = involved_counties(occ_df, workspace_root)
    area = load_display_area_wgs84(workspace_root, county_names)
    lims = wgs84_axis_limits(area)
    return area, lims, workspace_root
