"""Beijing administrative boundary (WGS84): download, cache, point-in-polygon.

Uses Aliyun DataV GeoJSON for municipality adcode 110000 (北京市界).
Source: https://geo.datav.aliyun.com/areas_v3/bound/110000.json
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_BEIJING_GEOJSON_URL = "https://geo.datav.aliyun.com/areas_v3/bound/110000.json"
DEFAULT_CACHE_FILENAME = "beijing_adcode_110000.json"


def download_beijing_boundary_json(
    cache_path: Path,
    *,
    url: str = DEFAULT_BEIJING_GEOJSON_URL,
    timeout: int = 120,
) -> None:
    import urllib.request

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "geolife-experiment/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        cache_path.write_bytes(resp.read())


def load_beijing_geometry(
    cache_dir: Path,
    *,
    url: str = DEFAULT_BEIJING_GEOJSON_URL,
    cache_filename: str = DEFAULT_CACHE_FILENAME,
) -> Any:
    from shapely.geometry import shape
    from shapely.ops import unary_union

    cache_path = cache_dir / cache_filename
    if not cache_path.is_file():
        download_beijing_boundary_json(cache_path, url=url)
    fc = json.loads(cache_path.read_text(encoding="utf-8"))
    geoms = [shape(f["geometry"]) for f in fc["features"]]
    return unary_union(geoms) if len(geoms) > 1 else geoms[0]


def points_in_beijing_polygon(lat: Any, lon: Any, geometry: Any) -> np.ndarray:
    """True if (lon, lat) lies inside the Beijing municipality polygon(s)."""
    from shapely import contains_xy

    la = np.asarray(lat, dtype=np.float64)
    lo = np.asarray(lon, dtype=np.float64)
    return np.asarray(contains_xy(geometry, lo, la), dtype=bool)
