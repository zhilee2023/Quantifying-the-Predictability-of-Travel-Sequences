from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable

import numpy as np
import pandas as pd
from pyproj import Transformer

from ctw_estimate import CTWEntropy

EARTH_RADIUS_M = 6_378_137.0
MERCATOR_LAT_LIMIT = 85.05112878
DEFAULT_KNOWN_MODES = {
    "airplane",
    "bike",
    "boat",
    "bus",
    "car",
    "motorcycle",
    "run",
    "subway",
    "taxi",
    "train",
    "walk",
}
SF_POINT_PATTERN = re.compile(r"POINT\(([-0-9.]+)\s+([-0-9.]+)\)")


def safe_log2(x: float, eps: float = 1e-20) -> float:
    return float(np.log2(max(x, eps)))


def f_e(p: float, alphabet_size: int) -> float:
    if p <= 0.0 or p >= 1.0:
        binary_entropy = 0.0
    else:
        binary_entropy = -p * safe_log2(p) - (1.0 - p) * safe_log2(1.0 - p)
    return binary_entropy + p * safe_log2(alphabet_size - 1)


def find_p(target_fe: float, alphabet_size: int, tol: float = 1e-10, max_iter: int = 200) -> float:
    if alphabet_size <= 1:
        return 1.0

    low = 1e-12
    high = 1.0 - 1e-12
    target_fe = float(np.clip(target_fe, f_e(0.0, alphabet_size), f_e(1.0, alphabet_size)))
    for _ in range(max_iter):
        mid = 0.5 * (low + high)
        value = f_e(mid, alphabet_size)
        if abs(value - target_fe) <= tol:
            return 1.0 - mid
        if value < target_fe:
            low = mid
        else:
            high = mid
    return 1.0 - 0.5 * (low + high)


@dataclass(frozen=True)
class SegmentSummary:
    user_id: str
    trajectory_id: str
    segment_id: str
    sample_interval_min: int
    interpolation_method: str
    dominant_mode: str
    num_points: int


def parse_wkt_point(point_text: str) -> tuple[float, float]:
    point = str(point_text).strip()
    match = SF_POINT_PATTERN.fullmatch(point)
    if match is None:
        return float("nan"), float("nan")
    longitude = float(match.group(1))
    latitude = float(match.group(2))
    return latitude, longitude


def load_sf_segment_table(data_path: str | Path) -> pd.DataFrame:
    path = Path(data_path)
    print(f"[sf] reading {path} ...", flush=True)
    df = pd.read_csv(path)
    print(f"[sf] loaded {len(df)} rows from CSV", flush=True)
    required = {"trajectory", "timestamp", "start_point", "end_point"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"SF dataset is missing required columns: {sorted(missing)}")

    df = df.copy()
    df["trajectory"] = df["trajectory"].astype(str).str.strip()
    df["user_id"] = df["trajectory"].str.extract(r"^([^_]+)_", expand=False).fillna(df["trajectory"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    start_points = df["start_point"].map(parse_wkt_point)
    df["latitude"] = start_points.map(lambda item: item[0])
    df["longitude"] = start_points.map(lambda item: item[1])
    df["altitude_ft"] = np.nan
    df = df.dropna(subset=["user_id", "timestamp", "latitude", "longitude"]).copy()
    df = df.sort_values(["user_id", "timestamp", "trajectory"]).reset_index(drop=True)
    return df[
        [
            "user_id",
            "trajectory",
            "timestamp",
            "latitude",
            "longitude",
            "altitude_ft",
            "start_point",
            "end_point",
        ]
    ]


def latlon_to_web_mercator_km(latitudes: Iterable[float], longitudes: Iterable[float]) -> tuple[np.ndarray, np.ndarray]:
    lat = np.clip(np.asarray(list(latitudes), dtype=float), -MERCATOR_LAT_LIMIT, MERCATOR_LAT_LIMIT)
    lon = np.asarray(list(longitudes), dtype=float)
    lat_rad = np.deg2rad(lat)
    lon_rad = np.deg2rad(lon)
    x_km = (EARTH_RADIUS_M * lon_rad) / 1000.0
    y_km = (EARTH_RADIUS_M * np.log(np.tan(np.pi / 4.0 + lat_rad / 2.0))) / 1000.0
    return x_km, y_km


def utm_epsg_from_latlon(latitude: float, longitude: float) -> int:
    """
    EPSG code for WGS 84 / UTM zone (northern: 32601–32660, southern: 32701–32760).
    Zone follows standard UTM: floor((lon + 180) / 6) + 1.
    """
    zone = int((longitude + 180.0) // 6) + 1
    zone = max(1, min(60, zone))
    if latitude >= 0.0:
        return 32600 + zone
    return 32700 + zone


def latlon_to_utm_km(
    latitudes: Iterable[float],
    longitudes: Iterable[float],
    epsg: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Project WGS84 (lat, lon) degrees to UTM easting/northing in **kilometers**.

    One UTM zone is used for the whole sequence (default: zone from the median lat/lon),
    so local trajectories lie in a linear metric space suitable for RMSE in km.
    """
    lat = np.asarray(list(latitudes), dtype=float)
    lon = np.asarray(list(longitudes), dtype=float)
    if lat.size == 0:
        return np.array([]), np.array([])
    if epsg is None:
        epsg = utm_epsg_from_latlon(float(np.median(lat)), float(np.median(lon)))
    transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    x_m, y_m = transformer.transform(lon, lat)
    return (x_m / 1000.0).astype(np.float64), (y_m / 1000.0).astype(np.float64)


def add_projected_coordinates(df: pd.DataFrame, projection: str = "utm") -> pd.DataFrame:
    """
    Add metric planar coordinates in km.

    Parameters
    ----------
    projection
        ``utm`` (default): WGS84 → UTM, consistent with RMSE in kilometers for distortion.
        ``web_mercator``: legacy Web Mercator–style x/y in km (not equal-area).
    """
    if df.empty:
        return df.copy()
    projected = df.copy()
    if projection == "utm":
        x_km, y_km = latlon_to_utm_km(projected["latitude"], projected["longitude"])
    elif projection == "web_mercator":
        x_km, y_km = latlon_to_web_mercator_km(projected["latitude"], projected["longitude"])
    else:
        raise ValueError("projection must be 'utm' or 'web_mercator'")
    projected["x_km"] = x_km
    projected["y_km"] = y_km
    projected["planar_projection"] = projection
    return projected


def assign_modes_to_timestamps(timestamps: pd.Series, labels: pd.DataFrame) -> tuple[list[str | None], int]:
    if labels.empty:
        return [None] * len(timestamps), 0

    assignments: list[set[str]] = [set() for _ in range(len(timestamps))]
    ts_values = timestamps.to_numpy(dtype="datetime64[ns]")

    for row in labels.itertuples(index=False):
        start = np.datetime64(row.start_time)
        end = np.datetime64(row.end_time)
        mask = (ts_values >= start) & (ts_values <= end)
        for index in np.flatnonzero(mask):
            assignments[index].add(row.mode)

    modes: list[str | None] = []
    ambiguous = 0
    for active_modes in assignments:
        if not active_modes:
            modes.append(None)
        elif len(active_modes) == 1:
            modes.append(next(iter(active_modes)))
        else:
            modes.append(None)
            ambiguous += 1
    return modes, ambiguous


def _build_target_index(
    timestamps: pd.Series,
    sample_interval_min: int,
) -> pd.DatetimeIndex:
    freq = f"{sample_interval_min}min"
    start = timestamps.iloc[0].ceil(freq)
    end = timestamps.iloc[-1].floor(freq)
    if start > end:
        return pd.DatetimeIndex([])
    return pd.date_range(start=start, end=end, freq=freq)


def resample_trajectory(
    trajectory_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    user_id: str,
    trajectory_id: str,
    sample_interval_min: int,
    interpolation_method: str,
    require_mode: bool = True,
    projection: str = "utm",
) -> pd.DataFrame:
    if trajectory_df.empty:
        return pd.DataFrame()

    projected = add_projected_coordinates(trajectory_df, projection=projection)
    projected = projected.sort_values("timestamp").drop_duplicates(subset=["timestamp"])
    target_index = _build_target_index(projected["timestamp"], sample_interval_min)
    if target_index.empty:
        return pd.DataFrame()

    source = projected.set_index("timestamp")[["latitude", "longitude", "x_km", "y_km", "altitude_ft"]].sort_index()

    if interpolation_method == "linear":
        union_index = source.index.union(target_index)
        aligned = source.reindex(union_index).sort_index().interpolate(method="time")
        resampled = aligned.loc[target_index].reset_index(names="timestamp")
    elif interpolation_method == "nearest":
        target = pd.DataFrame({"timestamp": target_index})
        observed = source.reset_index()
        resampled = pd.merge_asof(
            target,
            observed,
            on="timestamp",
            direction="nearest",
            tolerance=pd.Timedelta(minutes=sample_interval_min),
        )
        resampled = resampled.dropna(subset=["latitude", "longitude", "x_km", "y_km"]).reset_index(drop=True)
    else:
        raise ValueError("interpolation_method must be 'linear' or 'nearest'")

    if resampled.empty:
        return pd.DataFrame()

    modes, ambiguous_count = assign_modes_to_timestamps(resampled["timestamp"], labels_df)
    resampled["mode"] = modes
    resampled["user_id"] = user_id
    resampled["trajectory_id"] = trajectory_id
    resampled["sample_interval_min"] = int(sample_interval_min)
    resampled["interpolation_method"] = interpolation_method
    resampled["ambiguous_label_count"] = int(ambiguous_count)
    resampled["planar_projection"] = projection

    if require_mode:
        resampled = resampled.dropna(subset=["mode"]).reset_index(drop=True)

    if resampled.empty:
        return pd.DataFrame()
    return resampled


def split_contiguous_segments(
    sampled_df: pd.DataFrame,
    sample_interval_min: int,
    min_points: int = 5,
) -> pd.DataFrame:
    if sampled_df.empty:
        return pd.DataFrame()

    freq = pd.Timedelta(minutes=sample_interval_min)
    segmented = sampled_df.sort_values("timestamp").copy()
    time_gap = segmented["timestamp"].diff().gt(freq)
    missing_mode = segmented["mode"].isna() if "mode" in segmented.columns else pd.Series(False, index=segmented.index)
    new_segment = (time_gap | missing_mode).fillna(True)
    segmented["segment_index"] = new_segment.cumsum().astype(int)

    outputs = []
    for segment_idx, group in segmented.groupby("segment_index", sort=True):
        if len(group) < min_points:
            continue
        piece = group.drop(columns=["segment_index"]).copy()
        piece["segment_id"] = f"{piece['trajectory_id'].iloc[0]}_seg{segment_idx:03d}"
        outputs.append(piece)

    if not outputs:
        return pd.DataFrame()
    return pd.concat(outputs, ignore_index=True)


def build_sf_resampled_dataset(
    data_path: str | Path,
    sample_intervals: Iterable[int],
    interpolation_methods: Iterable[str],
    require_mode: bool = True,
    min_points: int = 5,
    max_users: int | None = None,
    selected_user_ids: Iterable[str] | None = None,
    projection: str = "utm",
) -> pd.DataFrame:
    sf_rows = load_sf_segment_table(data_path)
    records = []
    user_ids = sorted(sf_rows["user_id"].astype(str).unique().tolist())
    if selected_user_ids is not None:
        selected = {str(user_id) for user_id in selected_user_ids}
        user_ids = [user_id for user_id in user_ids if user_id in selected]
    if max_users is not None:
        user_ids = user_ids[:max_users]

    n_users = len(user_ids)
    print(f"[sf] resampling {n_users} vehicles × {list(sample_intervals)} min × {list(interpolation_methods)} ...", flush=True)
    for idx, user_id in enumerate(user_ids):
        if idx == 0 or (idx + 1) % 100 == 0 or idx + 1 == n_users:
            print(f"[sf] vehicle {idx + 1}/{n_users} ({user_id}) ...", flush=True)
        trajectory_df = (
            sf_rows.loc[sf_rows["user_id"] == user_id, ["timestamp", "latitude", "longitude", "altitude_ft"]]
            .sort_values("timestamp")
            .drop_duplicates(subset=["timestamp"])
            .reset_index(drop=True)
        )
        if len(trajectory_df) < 2:
            continue
        labels_df = pd.DataFrame(
            [
                {
                    "start_time": trajectory_df["timestamp"].min(),
                    "end_time": trajectory_df["timestamp"].max(),
                    "mode": "taxi",
                }
            ]
        )
        trajectory_id = str(user_id)
        for sample_interval in sample_intervals:
            for interpolation_method in interpolation_methods:
                resampled = resample_trajectory(
                    trajectory_df=trajectory_df,
                    labels_df=labels_df,
                    user_id=str(user_id),
                    trajectory_id=trajectory_id,
                    sample_interval_min=sample_interval,
                    interpolation_method=interpolation_method,
                    require_mode=require_mode,
                    projection=projection,
                )
                segments = split_contiguous_segments(
                    sampled_df=resampled,
                    sample_interval_min=sample_interval,
                    min_points=min_points,
                )
                if not segments.empty:
                    segments["source_file"] = str(data_path)
                    records.append(segments)

    if not records:
        return pd.DataFrame(
            columns=[
                "timestamp",
                "latitude",
                "longitude",
                "x_km",
                "y_km",
                "altitude_ft",
                "mode",
                "user_id",
                "trajectory_id",
                "sample_interval_min",
                "interpolation_method",
                "ambiguous_label_count",
                "segment_id",
                "source_file",
            ]
        )
    combined = pd.concat(records, ignore_index=True)
    combined = combined.sort_values(["sample_interval_min", "interpolation_method", "user_id", "segment_id", "timestamp"])
    return combined.reset_index(drop=True)


def extract_segment_arrays(samples_df: pd.DataFrame) -> list[dict[str, object]]:
    segments = []
    if samples_df.empty:
        return segments

    for segment_id, group in samples_df.groupby("segment_id", sort=True):
        ordered = group.sort_values("timestamp").reset_index(drop=True)
        xy = ordered[["x_km", "y_km"]].to_numpy(dtype=np.float32)
        modes = ordered["mode"].astype(str).tolist()
        summary = SegmentSummary(
            user_id=str(ordered["user_id"].iloc[0]),
            trajectory_id=str(ordered["trajectory_id"].iloc[0]),
            segment_id=str(segment_id),
            sample_interval_min=int(ordered["sample_interval_min"].iloc[0]),
            interpolation_method=str(ordered["interpolation_method"].iloc[0]),
            dominant_mode=_safe_dominant_mode(modes),
            num_points=len(ordered),
        )
        segments.append(
            {
                "summary": summary,
                "data": ordered,
                "xy_km": xy,
                "modes": modes,
            }
        )
    return segments


def _safe_dominant_mode(modes: Iterable[str]) -> str:
    filtered = [mode for mode in modes if mode]
    if not filtered:
        return "unknown"
    counter = Counter(filtered)
    return counter.most_common(1)[0][0]


def compute_radius_of_gyration_km(xy_km: np.ndarray) -> float:
    if len(xy_km) == 0:
        return float("nan")
    center = xy_km.mean(axis=0, keepdims=True)
    return float(np.sqrt(np.mean(np.sum((xy_km - center) ** 2, axis=1))))


def compute_step_metrics(xy_km: np.ndarray, sample_interval_min: int) -> tuple[np.ndarray, np.ndarray]:
    if len(xy_km) == 0:
        return np.array([], dtype=float), np.array([], dtype=float)
    deltas = np.diff(xy_km, axis=0, prepend=xy_km[[0]])
    step_km = np.sqrt(np.sum(deltas ** 2, axis=1))
    hours = sample_interval_min / 60.0
    speed_kmh = step_km / hours if hours > 0 else np.zeros_like(step_km)
    return step_km, speed_kmh


def grid_discretize(
    xy_km: np.ndarray,
    cell_size_km: float,
) -> dict[str, np.ndarray | float | int]:
    if len(xy_km) == 0:
        return {
            "tokens": np.array([], dtype=int),
            "cell_x": np.array([], dtype=int),
            "cell_y": np.array([], dtype=int),
            "centroids_km": np.empty((0, 2), dtype=float),
            "distortion_km": np.array([], dtype=float),
            "alphabet_size": 0,
        }

    cell_x = np.floor(xy_km[:, 0] / cell_size_km).astype(int)
    cell_y = np.floor(xy_km[:, 1] / cell_size_km).astype(int)
    mapping: dict[tuple[int, int], int] = {}
    next_token = 0
    tokens = np.zeros(len(xy_km), dtype=int)
    for index, cell in enumerate(zip(cell_x, cell_y)):
        if cell not in mapping:
            mapping[cell] = next_token
            next_token += 1
        tokens[index] = mapping[cell]

    centroids = np.column_stack(((cell_x + 0.5) * cell_size_km, (cell_y + 0.5) * cell_size_km))
    distortion = np.sqrt(np.sum((xy_km - centroids) ** 2, axis=1))
    return {
        "tokens": tokens,
        "cell_x": cell_x,
        "cell_y": cell_y,
        "centroids_km": centroids,
        "distortion_km": distortion,
        "alphabet_size": next_token,
    }


def symbolic_ctw_metrics(sequence: Iterable[int]) -> dict[str, float]:
    sequence = np.asarray(list(sequence), dtype=int)
    if sequence.size < 2:
        return {"entropy_rate_bits": float("nan"), "predictability": float("nan"), "alphabet_size": float("nan")}

    alphabet_size = int(sequence.max()) + 1
    entropy_rate = CTWEntropy(max_symbol=alphabet_size).calculate_entropy_rate(sequence.tolist())
    predictability = find_p(entropy_rate, alphabet_size)
    return {
        "entropy_rate_bits": float(entropy_rate),
        "predictability": float(predictability),
        "alphabet_size": float(alphabet_size),
    }


def markov_entropy_rate(sequence: Iterable[int], order: int, alpha: float = 1.0) -> float:
    seq = np.asarray(list(sequence), dtype=int)
    if seq.size <= order:
        return float("nan")

    alphabet_size = int(seq.max()) + 1
    transitions: defaultdict[tuple[int, ...], np.ndarray] = defaultdict(lambda: np.zeros(alphabet_size, dtype=float))
    for idx in range(order, len(seq)):
        context = tuple(seq[idx - order : idx])
        transitions[context][seq[idx]] += 1.0

    total_log_prob = 0.0
    count = 0
    for idx in range(order, len(seq)):
        context = tuple(seq[idx - order : idx])
        counts = transitions[context]
        probs = (counts + alpha) / (counts.sum() + alpha * alphabet_size)
        total_log_prob += -np.log2(probs[seq[idx]])
        count += 1

    return float(total_log_prob / max(count, 1))


def markov_metrics(sequence: Iterable[int], order: int, alpha: float = 1.0) -> dict[str, float]:
    entropy_rate = markov_entropy_rate(sequence=sequence, order=order, alpha=alpha)
    if np.isnan(entropy_rate):
        return {"entropy_rate_bits": float("nan"), "predictability": float("nan")}
    alphabet_size = int(np.max(np.asarray(list(sequence), dtype=int))) + 1
    return {
        "entropy_rate_bits": float(entropy_rate),
        "predictability": float(find_p(entropy_rate, alphabet_size)),
    }


def compute_dataset_metadata(samples_df: pd.DataFrame) -> dict[str, object]:
    if samples_df.empty:
        return {
            "num_rows": 0,
            "num_segments": 0,
            "num_users": 0,
            "modes": [],
        }
    modes = sorted(set(samples_df["mode"].dropna().astype(str)) | DEFAULT_KNOWN_MODES.intersection(set(samples_df["mode"].dropna().astype(str))))
    return {
        "num_rows": int(len(samples_df)),
        "num_segments": int(samples_df["segment_id"].nunique()),
        "num_users": int(samples_df["user_id"].nunique()),
        "modes": modes,
    }
