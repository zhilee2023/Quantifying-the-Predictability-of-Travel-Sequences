from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
try:
    import torch
    from torch.utils.data import DataLoader
except ImportError:
    torch = None
    DataLoader = None

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from sf_preprocess import (
    build_sf_resampled_dataset,
    compute_dataset_metadata,
    compute_radius_of_gyration_km,
    compute_step_metrics,
    extract_segment_arrays,
    grid_discretize,
    markov_metrics,
    symbolic_ctw_metrics,
)

# sf_cabspotting package root (parent of src/)
WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SF_DATA_PATH = WORKSPACE_ROOT / "data" / "sf_dataset.csv"
DEFAULT_OUTPUT_DIR = WORKSPACE_ROOT / "sfexp_result"


def sliding_window_batches(X: np.ndarray, time_steps: int, stride: int = 1) -> np.ndarray:
    length, feature_dim = X.shape
    if length < time_steps:
        return np.empty((0, time_steps, feature_dim), dtype=X.dtype)
    windows = []
    for start in range(0, length - time_steps + 1, stride):
        windows.append(X[start : start + time_steps])
    return np.stack(windows, axis=0) if windows else np.empty((0, time_steps, feature_dim), dtype=X.dtype)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run trajectory predictability experiments.")
    parser.add_argument("--dataset", choices=["sf"], default="sf", help="Cabspotting / San Francisco only in this bundle.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_SF_DATA_PATH,
        help="Cabspotting-style CSV (see sf_cabspotting/data/README.md).",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sample-intervals", nargs="+", type=int, default=[5, 15, 30])
    parser.add_argument("--interpolation-methods", nargs="+", default=["linear", "nearest"])
    parser.add_argument("--codebook-sizes", nargs="+", type=int, default=[1024, 2048, 4096])
    parser.add_argument(
        "--tolerance-kms",
        nargs="+",
        type=float,
        default=[2.5, 5.0, 7.5, 10.0],
        help="Grid cell widths (km) for direct_ctw / Markov and VQ D_target; one scenario subfolder per (tolerance × interval).",
    )
    parser.add_argument(
        "--tolerance-km",
        type=float,
        default=None,
        help="If set, overrides --tolerance-kms to this single value (e.g. for quick tests).",
    )
    parser.add_argument(
        "--coordinate-scale-km",
        type=float,
        default=10.0,
        help="Divides centered x/y (km) before VQ; must match training and eval. See --tolerance-km for D_target.",
    )
    parser.add_argument("--time-steps", type=int, default=32)
    parser.add_argument("--window-stride", type=int, default=4)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Training batch size (lower default helps large K and GPU memory).",
    )
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--num-epochs", type=int, default=80)
    parser.add_argument(
        "--pretrain-epochs",
        type=int,
        default=5,
        help="VQ-only warmup before augmented Lagrangian (recon <= D_target).",
    )
    parser.add_argument(
        "--hidden-channels",
        type=int,
        default=64,
        help="Conv trunk width (slightly wider default for large codebooks).",
    )
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=16,
        help="VQ embedding dimension (larger default pairs better with K>=1024).",
    )
    parser.add_argument("--num-conv-layers", type=int, default=3)
    parser.add_argument("--kernel-size", type=int, default=13)
    parser.add_argument(
        "--commitment-cost",
        type=float,
        default=0.35,
        help="VQ commitment beta (slightly higher default to reduce unused codes at large K).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=3e-4,
        help="Adam LR (conservative default for large codebook softmax + ALM).",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=1.08,
        help="ALM: multiplicative factor for rho when recon violates D_target (gentler than 1.1).",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.5,
        help="StepLR: multiply LR by this every --step-size epochs (less aggressive decay).",
    )
    parser.add_argument(
        "--step-size",
        type=int,
        default=15,
        help="StepLR: decay LR every N epochs (after pretrain).",
    )
    parser.add_argument("--markov-alpha", type=float, default=1.0)
    parser.add_argument("--min-points", type=int, default=5)
    parser.add_argument("--min-vq-length", type=int, default=32)
    parser.add_argument("--max-users", type=int, default=None)
    parser.add_argument("--max-segments", type=int, default=None)
    parser.add_argument(
        "--projection",
        choices=["utm", "web_mercator"],
        default="utm",
        help="Planar metric for x_km/y_km: UTM (recommended, RMSE in km) or legacy Web Mercator.",
    )
    parser.add_argument("--device", default="cuda" if torch is not None and torch.cuda.is_available() else "cpu")
    parser.add_argument("--skip-vq", action="store_true")
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="If set, write outputs under output-dir/run-name instead of a timestamped folder (useful on HPC).",
    )
    parser.add_argument(
        "--hpc-coarse-task",
        type=int,
        default=None,
        metavar="0..5",
        help="Parallelism: run exactly one (sample_interval × interpolation) pair. "
        "Task id 0..5 maps to 3 intervals × 2 methods (all codebook sizes in-process). "
        "Mutually exclusive with --hpc-fine-task.",
    )
    parser.add_argument(
        "--hpc-fine-task",
        type=int,
        default=None,
        metavar="0..17",
        help="Max parallelism: run exactly one (interval × interpolation × codebook K) triple. "
        "Task id 0..17 = 3×2×3 grid. Recomputes baselines each job (merge/dedupe downstream). "
        "Mutually exclusive with --hpc-coarse-task.",
    )
    parser.add_argument(
        "--hpc-scenario-task",
        type=int,
        default=None,
        metavar="0..11",
        help="Parallelism: run exactly one (tolerance_km × sample_interval) pair from the default 4×3 grid "
        "(2.5/5/7.5/10 km × 5/15/30 min). Task id = 3*tolerance_index + interval_index (tolerance 0..3, interval 0..2). "
        "Mutually exclusive with --hpc-coarse-task and --hpc-fine-task.",
    )
    return parser.parse_args()


def _apply_hpc_scenario_task(args: argparse.Namespace) -> None:
    """One SLURM array id per (tolerance × sampling interval) from the default 4×3 grid."""
    if args.hpc_scenario_task is None:
        return
    if args.hpc_coarse_task is not None or args.hpc_fine_task is not None:
        raise SystemExit("--hpc-scenario-task cannot be combined with --hpc-coarse-task or --hpc-fine-task.")
    tolerance_grid = [2.5, 5.0, 7.5, 10.0]
    interval_grid = [5, 15, 30]
    idx = int(args.hpc_scenario_task)
    if not (0 <= idx < len(tolerance_grid) * len(interval_grid)):
        raise SystemExit("--hpc-scenario-task must be in 0..11")
    tol_idx = idx // len(interval_grid)
    int_idx = idx % len(interval_grid)
    args.tolerance_kms = [tolerance_grid[tol_idx]]
    args.sample_intervals = [interval_grid[int_idx]]


def _apply_hpc_task_slices(args: argparse.Namespace) -> None:
    """Restrict the Cartesian grid for multi-GPU / array jobs."""
    intervals_default = [5, 15, 30]
    interps_default = ["linear", "nearest"]
    ks_default = [1024, 2048, 4096]

    if args.hpc_scenario_task is not None:
        return

    if args.hpc_coarse_task is not None and args.hpc_fine_task is not None:
        raise SystemExit("Use only one of --hpc-coarse-task or --hpc-fine-task.")

    if args.hpc_coarse_task is not None:
        idx = int(args.hpc_coarse_task)
        if not (0 <= idx < 6):
            raise SystemExit("--hpc-coarse-task must be in 0..5")
        si = idx // 2
        mi = idx % 2
        args.sample_intervals = [intervals_default[si]]
        args.interpolation_methods = [interps_default[mi]]

    if args.hpc_fine_task is not None:
        idx = int(args.hpc_fine_task)
        if not (0 <= idx < 18):
            raise SystemExit("--hpc-fine-task must be in 0..17")
        ki = idx % 3
        t = idx // 3
        mi = t % 2
        si = t // 2
        args.sample_intervals = [intervals_default[si]]
        args.interpolation_methods = [interps_default[mi]]
        args.codebook_sizes = [ks_default[ki]]


def resolve_output_dir(base_dir: Path, run_name: str | None) -> Path:
    if run_name:
        safe = run_name.strip().replace("..", "_")
        run_dir = base_dir / safe
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir
    run_dir = base_dir / datetime.now().strftime("sf_run_%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def write_json(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=_json_default)


def _json_default(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Unsupported JSON value: {type(value)}")


def reindex_symbols(sequence: Iterable[int]) -> np.ndarray:
    seq = np.asarray(list(sequence), dtype=int)
    if seq.size == 0:
        return seq
    mapping: dict[int, int] = {}
    next_token = 0
    reindexed = np.zeros_like(seq)
    for idx, token in enumerate(seq):
        if token not in mapping:
            mapping[token] = next_token
            next_token += 1
        reindexed[idx] = mapping[token]
    return reindexed


def weighted_average(series: pd.Series, weights: pd.Series) -> float:
    valid = ~(series.isna() | weights.isna())
    if not valid.any():
        return float("nan")
    return float(np.average(series[valid], weights=weights[valid]))


def aggregate_weighted(
    df: pd.DataFrame,
    group_cols: list[str],
    value_cols: list[str],
    weight_col: str = "num_points",
) -> pd.DataFrame:
    rows = []
    for keys, group in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        row[weight_col] = float(group[weight_col].sum())
        for column in value_cols:
            row[column] = weighted_average(group[column], group[weight_col])
        rows.append(row)
    return pd.DataFrame(rows)


def _method_column_key(row: pd.Series) -> str:
    """Stable key for pivot: direct_ctw, markov_order_1, vq_ctw_K2048, ..."""
    m = str(row["method"])
    if m == "vq_ctw" and pd.notna(row.get("codebook_size")):
        return f"vq_ctw_K{int(row['codebook_size'])}"
    return m


def save_user_predictability_exports(user_df: pd.DataFrame, scenario_dir: Path) -> None:
    """
    Wide table: one row per (user, interval, interpolation, tolerance), columns for each method's
    predictability and entropy_rate_bits (direct_ctw, markov_order_*, vq_ctw_K*, ...).
    """
    if user_df.empty:
        return
    long_sorted = user_df.sort_values(["user_id", "method", "codebook_size"], na_position="last")
    long_sorted.to_csv(scenario_dir / "user_predictability_all_methods.csv", index=False)

    df = long_sorted.copy()
    df["_method_key"] = df.apply(_method_column_key, axis=1)
    id_cols = ["user_id", "sample_interval_min", "interpolation_method", "tolerance_km"]
    pred_pt = df.pivot_table(index=id_cols, columns="_method_key", values="predictability", aggfunc="first")
    ent_pt = df.pivot_table(index=id_cols, columns="_method_key", values="entropy_rate_bits", aggfunc="first")
    pred_pt.columns = [f"predictability_{c}" for c in pred_pt.columns]
    ent_pt.columns = [f"entropy_rate_bits_{c}" for c in ent_pt.columns]
    wide = pred_pt.join(ent_pt, how="outer").reset_index()
    wide.to_csv(scenario_dir / "user_predictability_wide.csv", index=False)


def save_user_label_exports(samples_df: pd.DataFrame, output_dir: Path) -> None:
    """
    Persist mode labels for Cabspotting/SF: resampled counts, per-user mode fractions,
    and one summary interval per user (min/max timestamp, mode taxi).
    """
    if samples_df.empty:
        return
    long_cnt = (
        samples_df.groupby(
            ["user_id", "sample_interval_min", "interpolation_method", "mode"],
            dropna=False,
        )
        .size()
        .reset_index(name="num_points")
    )
    long_cnt.to_csv(output_dir / "user_resampled_label_mode_counts.csv", index=False)

    wide = long_cnt.pivot_table(
        index=["user_id", "sample_interval_min", "interpolation_method"],
        columns="mode",
        values="num_points",
        fill_value=0,
        aggfunc="sum",
    )
    wide.reset_index().to_csv(output_dir / "user_resampled_label_mode_counts_wide.csv", index=False)

    frac = samples_df.groupby(["user_id", "mode"]).size().reset_index(name="num_points")
    tot = frac.groupby("user_id")["num_points"].transform("sum")
    frac["fraction_of_user_points"] = frac["num_points"] / tot.replace(0, np.nan)
    frac.to_csv(output_dir / "user_label_mode_fraction_by_user.csv", index=False)

    intervals = (
        samples_df.groupby("user_id", as_index=False)
        .agg(start_time=("timestamp", "min"), end_time=("timestamp", "max"))
        .assign(mode="taxi")
    )
    intervals["start_time"] = intervals["start_time"].map(lambda value: value.isoformat() if pd.notna(value) else "")
    intervals["end_time"] = intervals["end_time"].map(lambda value: value.isoformat() if pd.notna(value) else "")
    intervals.to_csv(output_dir / "user_original_label_intervals.csv", index=False)


def concat_all_scenarios_user_predictability(output_dir: Path, args: argparse.Namespace) -> None:
    """Merge per-scenario user-level tables into one CSV under the run root."""
    frames: list[pd.DataFrame] = []
    for tol_km in args.tolerance_kms:
        for sample_interval in args.sample_intervals:
            scenario_dir = output_dir / scenario_subdir_name(tol_km, sample_interval)
            path = scenario_dir / "user_predictability_all_methods.csv"
            if not path.is_file():
                path = scenario_dir / "user_level_results.csv"
            if not path.is_file():
                continue
            chunk = pd.read_csv(path)
            if "scenario_subdir" not in chunk.columns:
                chunk.insert(0, "scenario_subdir", scenario_dir.name)
            frames.append(chunk)
    if not frames:
        return
    merged = pd.concat(frames, ignore_index=True)
    merged.to_csv(output_dir / "all_scenarios_user_predictability_all_methods.csv", index=False)


def scenario_subdir_name(tolerance_km: float, sample_interval_min: int) -> str:
    """Folder name for one (spatial precision × resampling interval) scenario."""
    tol_str = f"{tolerance_km:g}".replace(".", "p")
    return f"tol_{tol_str}km_int_{sample_interval_min}min"


def vq_reconstruction_target_scaled(grid_cell_km: float, coordinate_scale_km: float) -> float:
    """
    D_target passed to model_train: mean squared Euclidean distance in **scaled** (x,y) coords.

    Here D_target = (grid_cell_km / coordinate_scale_km)^2, so sqrt(D_target) * coordinate_scale_km == grid_cell_km:
    the inequality recon_loss <= D_target is equivalent to RMSE (km) <= grid_cell_km when read against training logs.
    """
    return (grid_cell_km / coordinate_scale_km) ** 2


def _init_scenario_training_losses(
    scenario_dir: Path,
    tolerance_km: float,
    sample_interval: int,
    args: argparse.Namespace,
) -> None:
    """Create ``scenario_training_losses.txt`` header (one file per scenario, all K/interp sections appended)."""
    if args.skip_vq:
        text = "# VQ skipped (--skip-vq). No training losses.\n"
        (scenario_dir / "scenario_training_losses.txt").write_text(text, encoding="utf-8")
        return
    d_t = vq_reconstruction_target_scaled(tolerance_km, args.coordinate_scale_km)
    target_rmse_km = math.sqrt(max(d_t, 0.0)) * float(args.coordinate_scale_km)
    lines = [
        "# VQ-VAE training losses (per epoch). One block per (interval, interpolation, K).",
        f"# tolerance_km={tolerance_km}  sample_interval_min={sample_interval}",
        f"# coordinate_scale_km={args.coordinate_scale_km}  D_target_scaled={d_t}  target_RMSE_km={target_rmse_km:.6g}",
        "",
    ]
    (scenario_dir / "scenario_training_losses.txt").write_text("\n".join(lines), encoding="utf-8")


def _append_scenario_training_loss_section(
    scenario_dir: Path,
    *,
    sample_interval: int,
    interpolation_method: str,
    codebook_size: int,
    log_path: Path,
) -> None:
    """Append a training run's loss log into ``scenario_training_losses.txt``."""
    summary_path = scenario_dir / "scenario_training_losses.txt"
    sep = "=" * 80
    head = (
        f"\n{sep}\n"
        f"# sample_interval_min={sample_interval}  interpolation={interpolation_method}  K={codebook_size}\n"
        f"# detail_log: {log_path.name}\n"
        f"{sep}\n\n"
    )
    if log_path.is_file():
        body = log_path.read_text(encoding="utf-8", errors="replace")
    else:
        body = "(missing detail log file)\n"
    with summary_path.open("a", encoding="utf-8") as f:
        f.write(head + body)


def build_training_windows(
    segment_arrays: list[dict[str, object]],
    time_steps: int,
    stride: int,
    center_km: np.ndarray,
    scale_km: float,
    max_segments: int | None = None,
) -> np.ndarray:
    windows = []
    usable_segments = segment_arrays[:max_segments] if max_segments is not None else segment_arrays
    for segment in usable_segments:
        xy_km = np.asarray(segment["xy_km"], dtype=np.float32)
        if len(xy_km) < time_steps:
            continue
        xy_scaled = (xy_km - center_km) / scale_km
        windows.append(sliding_window_batches(xy_scaled, time_steps, stride=stride))

    if not windows:
        return np.empty((0, time_steps, 2), dtype=np.float32)
    return np.concatenate(windows, axis=0).astype(np.float32)


def build_window_starts(length: int, time_steps: int, step: int) -> list[int]:
    if length < time_steps:
        return []
    starts = list(range(0, length - time_steps + 1, step))
    last_start = length - time_steps
    if starts[-1] != last_start:
        starts.append(last_start)
    return starts


def train_vqvae(
    train_windows: np.ndarray,
    args: argparse.Namespace,
    codebook_size: int,
    d_target_scaled: float,
    log_path: Path,
) -> object:
    if torch is None or DataLoader is None:
        raise RuntimeError("PyTorch is required for VQ-VAE training. Install torch or run with --skip-vq.")

    from model import EC_VQVAE, model_train

    dataset = torch.tensor(train_windows, dtype=torch.float32)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=len(dataset) >= args.batch_size)
    model = EC_VQVAE(
        in_channels=2,
        hidden_channels=args.hidden_channels,
        codebook_size=codebook_size,
        embedding_dim=args.embedding_dim,
        commitment_cost=args.commitment_cost,
        time_steps=args.time_steps,
        num_conv_layers=args.num_conv_layers,
        kernel_size=args.kernel_size,
    ).to(args.device)
    model_train(
        model=model,
        dataloader=dataloader,
        device=torch.device(args.device),
        num_epochs=args.num_epochs,
        output_file=str(log_path),
        pretrain_epochs=args.pretrain_epochs,
        step_size=args.step_size,
        sigma=args.sigma,
        gamma=args.gamma,
        D_target=d_target_scaled,
        lr=args.lr,
        coordinate_scale_km=args.coordinate_scale_km,
    )
    return model


def vqvae_checkpoint_stem(
    tolerance_km: float,
    sample_interval_min: int,
    interpolation_method: str,
    codebook_size: int,
) -> str:
    tol_str = f"{tolerance_km:g}".replace(".", "p")
    safe_interp = str(interpolation_method)
    return f"vqvae_tol{tol_str}km_int{sample_interval_min}min_{safe_interp}_K{codebook_size}"


def save_vqvae_checkpoint(
    model: object,
    scenario_dir: Path,
    *,
    tolerance_km: float,
    sample_interval: int,
    interpolation_method: str,
    codebook_size: int,
    d_target_scaled: float,
    center_km: np.ndarray,
    args: argparse.Namespace,
) -> None:
    """Save weights + JSON metadata so EC_VQVAE can be reinstantiated for this (tol, interval, interp, K)."""
    if torch is None:
        return
    stem = vqvae_checkpoint_stem(tolerance_km, sample_interval, interpolation_method, codebook_size)
    pt_path = scenario_dir / f"{stem}.pt"
    meta_path = scenario_dir / f"{stem}_meta.json"

    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    torch.save(state, pt_path)

    center = np.asarray(center_km, dtype=float).reshape(-1).tolist()
    constructor_kwargs = {
        "in_channels": 2,
        "hidden_channels": args.hidden_channels,
        "codebook_size": int(codebook_size),
        "embedding_dim": args.embedding_dim,
        "commitment_cost": args.commitment_cost,
        "time_steps": args.time_steps,
        "num_conv_layers": args.num_conv_layers,
        "kernel_size": args.kernel_size,
    }
    meta = {
        "model_class": "EC_VQVAE",
        "module": "model",
        "constructor_kwargs": constructor_kwargs,
        "experiment": {
            "tolerance_km": float(tolerance_km),
            "sample_interval_min": int(sample_interval),
            "interpolation_method": str(interpolation_method),
            "coordinate_scale_km": float(args.coordinate_scale_km),
            "d_target_scaled": float(d_target_scaled),
            "center_km_xy": center,
            "device_trained": str(args.device),
            "num_epochs": int(args.num_epochs),
            "pretrain_epochs": int(args.pretrain_epochs),
            "batch_size": int(args.batch_size),
            "window_stride": int(args.window_stride),
            "lr": float(args.lr),
        },
        "checkpoint_file": pt_path.name,
    }
    write_json(meta_path, meta)


def encode_segment_stitched(
    model: object,
    xy_scaled: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, float]:
    step = max(1, args.time_steps - (args.kernel_size - 1))
    starts = build_window_starts(len(xy_scaled), args.time_steps, step)
    if not starts:
        return np.array([], dtype=int), np.empty((0, 2), dtype=np.float32), float("nan")

    windows = np.stack([xy_scaled[start : start + args.time_steps] for start in starts], axis=0)
    recon_sum = np.zeros_like(xy_scaled, dtype=np.float32)
    recon_count = np.zeros(len(xy_scaled), dtype=np.float32)
    codes = np.full(len(xy_scaled), -1, dtype=int)

    model.eval()
    with torch.no_grad():
        for batch_start in range(0, len(windows), args.eval_batch_size):
            batch_end = min(batch_start + args.eval_batch_size, len(windows))
            batch = torch.tensor(windows[batch_start:batch_end], dtype=torch.float32, device=args.device)
            x_recon, _, _, _, encoding_indices = model(batch)
            x_recon_np = x_recon.cpu().numpy()
            code_np = encoding_indices.cpu().numpy()

            for local_idx, global_idx in enumerate(range(batch_start, batch_end)):
                start = starts[global_idx]
                stop = start + args.time_steps
                recon_sum[start:stop] += x_recon_np[local_idx]
                recon_count[start:stop] += 1.0
                unassigned = codes[start:stop] < 0
                codes[start:stop][unassigned] = code_np[local_idx][unassigned]

    recon = recon_sum / np.maximum(recon_count[:, None], 1.0)
    for index in np.flatnonzero(codes < 0):
        codes[index] = 0

    recon_mse_scaled = float(np.mean(np.sum((recon - xy_scaled) ** 2, axis=1)))
    return codes, recon, recon_mse_scaled


def evaluate_baselines(
    segment: dict[str, object],
    cell_size_km: float,
    markov_alpha: float,
    tolerance_km: float,
) -> list[dict[str, object]]:
    summary = segment["summary"]
    xy_km = np.asarray(segment["xy_km"], dtype=np.float32)
    direct = grid_discretize(xy_km, cell_size_km)
    direct_tokens = direct["tokens"]
    direct_reindexed = reindex_symbols(direct_tokens)
    direct_ctw = symbolic_ctw_metrics(direct_reindexed)
    markov1 = markov_metrics(direct_reindexed, order=1, alpha=markov_alpha)
    markov2 = markov_metrics(direct_reindexed, order=2, alpha=markov_alpha)
    direct_distortion = np.asarray(direct["distortion_km"], dtype=float)
    baseline_common = {
        "user_id": summary.user_id,
        "trajectory_id": summary.trajectory_id,
        "segment_id": summary.segment_id,
        "sample_interval_min": summary.sample_interval_min,
        "interpolation_method": summary.interpolation_method,
        "tolerance_km": float(tolerance_km),
        "dominant_mode": summary.dominant_mode,
        "num_points": summary.num_points,
        "actual_distortion_km": float(np.mean(direct_distortion)) if direct_distortion.size else float("nan"),
        "reconstruction_rmse_km": float(np.sqrt(np.mean(direct_distortion**2))) if direct_distortion.size else float("nan"),
        "codebook_size": np.nan,
        "eligible_for_vq": len(xy_km) >= 1,
    }
    rows = []
    rows.append(
        {
            **baseline_common,
            "method": "direct_ctw",
            "entropy_rate_bits": direct_ctw["entropy_rate_bits"],
            "predictability": direct_ctw["predictability"],
            "alphabet_size": direct_ctw["alphabet_size"],
        }
    )
    rows.append(
        {
            **baseline_common,
            "method": "markov_order_1",
            "entropy_rate_bits": markov1["entropy_rate_bits"],
            "predictability": markov1["predictability"],
            "alphabet_size": direct_ctw["alphabet_size"],
        }
    )
    rows.append(
        {
            **baseline_common,
            "method": "markov_order_2",
            "entropy_rate_bits": markov2["entropy_rate_bits"],
            "predictability": markov2["predictability"],
            "alphabet_size": direct_ctw["alphabet_size"],
        }
    )
    return rows


def evaluate_vqvae(
    segment: dict[str, object],
    model: object,
    center_km: np.ndarray,
    scale_km: float,
    args: argparse.Namespace,
    codebook_size: int,
) -> tuple[dict[str, object], pd.DataFrame]:
    summary = segment["summary"]
    xy_km = np.asarray(segment["xy_km"], dtype=np.float32)
    if len(xy_km) < max(args.min_vq_length, args.time_steps):
        row = {
            "user_id": summary.user_id,
            "trajectory_id": summary.trajectory_id,
            "segment_id": summary.segment_id,
            "sample_interval_min": summary.sample_interval_min,
            "interpolation_method": summary.interpolation_method,
            "tolerance_km": float(args.tolerance_km),
            "dominant_mode": summary.dominant_mode,
            "num_points": summary.num_points,
            "actual_distortion_km": float("nan"),
            "reconstruction_rmse_km": float("nan"),
            "codebook_size": int(codebook_size),
            "eligible_for_vq": False,
            "method": "vq_ctw",
            "entropy_rate_bits": float("nan"),
            "predictability": float("nan"),
            "alphabet_size": float("nan"),
        }
        return row, pd.DataFrame()

    xy_scaled = (xy_km - center_km) / scale_km
    raw_codes, recon_scaled, recon_mse_scaled = encode_segment_stitched(model, xy_scaled, args)
    recon_km = recon_scaled * scale_km + center_km
    reindexed_codes = reindex_symbols(raw_codes)
    vq_metrics = symbolic_ctw_metrics(reindexed_codes)
    rmse_km = float(math.sqrt(recon_mse_scaled) * scale_km)
    radius_of_gyration = compute_radius_of_gyration_km(xy_km)
    step_km, speed_kmh = compute_step_metrics(xy_km, summary.sample_interval_min)

    row = {
        "user_id": summary.user_id,
        "trajectory_id": summary.trajectory_id,
        "segment_id": summary.segment_id,
        "sample_interval_min": summary.sample_interval_min,
        "interpolation_method": summary.interpolation_method,
        "tolerance_km": float(args.tolerance_km),
        "dominant_mode": summary.dominant_mode,
        "num_points": summary.num_points,
        "actual_distortion_km": rmse_km,
        "reconstruction_rmse_km": rmse_km,
        "codebook_size": int(codebook_size),
        "eligible_for_vq": True,
        "method": "vq_ctw",
        "entropy_rate_bits": vq_metrics["entropy_rate_bits"],
        "predictability": vq_metrics["predictability"],
        "alphabet_size": vq_metrics["alphabet_size"],
    }

    occurrence_df = segment["data"].copy()
    occurrence_df["tolerance_km"] = float(args.tolerance_km)
    occurrence_df["raw_code"] = raw_codes
    occurrence_df["reindexed_code"] = reindexed_codes
    occurrence_df["recon_x_km"] = recon_km[:, 0]
    occurrence_df["recon_y_km"] = recon_km[:, 1]
    occurrence_df["step_km"] = step_km
    occurrence_df["speed_kmh"] = speed_kmh
    occurrence_df["radius_of_gyration_km"] = radius_of_gyration
    occurrence_df["codebook_size"] = int(codebook_size)
    return row, occurrence_df


def summarize_latent_codes(occurrences_df: pd.DataFrame) -> pd.DataFrame:
    if occurrences_df.empty:
        return pd.DataFrame(
            columns=[
                "sample_interval_min",
                "interpolation_method",
                "codebook_size",
                "raw_code",
                "num_occurrences",
                "dominant_mode",
                "top_modes",
                "mean_speed_kmh",
                "mean_step_km",
                "mean_radius_of_gyration_km",
            ]
        )

    rows = []
    for keys, group in occurrences_df.groupby(["sample_interval_min", "interpolation_method", "codebook_size", "raw_code"], sort=True):
        interval, interpolation, codebook_size, raw_code = keys
        mode_counts = group["mode"].astype(str).value_counts()
        dominant_mode = mode_counts.index[0]
        top_modes = ", ".join(f"{mode}:{count}" for mode, count in mode_counts.head(3).items())
        rows.append(
            {
                "sample_interval_min": int(interval),
                "interpolation_method": str(interpolation),
                "codebook_size": int(codebook_size),
                "raw_code": int(raw_code),
                "num_occurrences": int(len(group)),
                "dominant_mode": dominant_mode,
                "top_modes": top_modes,
                "mean_speed_kmh": float(group["speed_kmh"].mean()),
                "mean_step_km": float(group["step_km"].mean()),
                "mean_radius_of_gyration_km": float(group["radius_of_gyration_km"].mean()),
            }
        )
    return pd.DataFrame(rows)


def save_summary_plots(results_df: pd.DataFrame, output_dir: Path) -> None:
    if results_df.empty:
        return

    summary = aggregate_weighted(
        df=results_df.dropna(subset=["predictability"]),
        group_cols=["method", "sample_interval_min", "interpolation_method", "codebook_size"],
        value_cols=["predictability", "reconstruction_rmse_km"],
    )

    plot_df = summary.copy()
    if plot_df.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    for method, group in plot_df.groupby("method", sort=True):
        label_suffix = ""
        if method == "vq_ctw":
            label_suffix = " (median K)"
            median_k = sorted(group["codebook_size"].dropna().astype(int).unique())
            if median_k:
                chosen_k = median_k[len(median_k) // 2]
                group = group[group["codebook_size"] == chosen_k]
        grouped = group.groupby("sample_interval_min")["predictability"].mean().reset_index()
        ax.plot(grouped["sample_interval_min"], grouped["predictability"], marker="o", label=f"{method}{label_suffix}")
    ax.set_xlabel("Sampling interval (minutes)")
    ax.set_ylabel("Weighted mean predictability")
    ax.set_title("Predictability vs. sampling interval")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "predictability_by_interval.png", dpi=200)
    plt.close(fig)

    vq_df = plot_df.loc[plot_df["method"] == "vq_ctw"].dropna(subset=["codebook_size"])
    if not vq_df.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        for interval, group in vq_df.groupby("sample_interval_min", sort=True):
            grouped = group.groupby("codebook_size")["predictability"].mean().reset_index()
            ax.plot(grouped["codebook_size"], grouped["predictability"], marker="o", label=f"{interval} min")
        ax.set_xlabel("Codebook size K")
        ax.set_ylabel("Weighted mean predictability")
        ax.set_title("Codebook sensitivity")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / "codebook_sensitivity.png", dpi=200)
        plt.close(fig)


def build_reviewer_summary(
    results_df: pd.DataFrame,
    code_summary_df: pd.DataFrame,
    metadata: dict[str, object],
    codebook_sizes: list[int],
) -> str:
    dataset_label = str(metadata.get("dataset_label", metadata.get("dataset", "Dataset")))
    modes = metadata.get("modes", [])
    lines = [
        f"# {dataset_label} Reviewer Summary",
        "",
        f"- Dataset: {dataset_label}, {metadata['num_users']} users, {metadata['num_segments']} resampled segments, {metadata['num_rows']} samples.",
        f"- Modes observed: {', '.join(modes) if modes else 'n/a'}.",
        "",
        "## Reviewer mapping",
        "",
    ]
    if metadata.get("skip_vq"):
        lines.append("- VQ-VAE execution was skipped in this run because PyTorch was unavailable or `--skip-vq` was set; baseline outputs remain valid.")
        lines.append("")

    main_codebook = sorted(codebook_sizes)[len(codebook_sizes) // 2]
    main_vq = results_df.loc[(results_df["method"] == "vq_ctw") & (results_df["codebook_size"] == main_codebook)]
    direct = results_df.loc[results_df["method"] == "direct_ctw"]
    markov1 = results_df.loc[results_df["method"] == "markov_order_1"]
    markov2 = results_df.loc[results_df["method"] == "markov_order_2"]

    def describe(df: pd.DataFrame) -> str:
        if df.empty:
            return "n/a"
        return f"{df['predictability'].mean():.4f}"

    lines.append(f"- `1.8`: compare `direct_ctw` ({describe(direct)}) against `vq_ctw` at `K={main_codebook}` ({describe(main_vq)}).")
    lines.append(f"- `2.3`: compare Markov baselines `order 1` ({describe(markov1)}) and `order 2` ({describe(markov2)}) against VQ-VAE.")
    lines.append("- `2.5`: compare `linear` and `nearest` interpolation rows in `segment_level_results.csv` and `user_level_results.csv`.")
    lines.append("- `2.6`: inspect `latent_code_summary.csv` for dominant modes, speed, step length, and radius-of-gyration profiles.")
    lines.append("- `2.7`: Dataset-specific run metadata is captured in `run_metadata.json` and the segment/user-level CSV outputs.")
    lines.append("- `3.3`: use `predictability_by_interval.png` and `codebook_sensitivity.png` for interval and codebook-size sensitivity.")
    lines.append("")
    lines.append("## Most common latent primitives")
    lines.append("")

    if code_summary_df.empty:
        lines.append("- No eligible VQ segments were long enough to summarize latent codes.")
    else:
        top_codes = code_summary_df.sort_values("num_occurrences", ascending=False).head(10)
        for row in top_codes.itertuples(index=False):
            lines.append(
                f"- Interval {row.sample_interval_min} min, K={row.codebook_size}, code {row.raw_code}: "
                f"{row.dominant_mode}, mean speed {row.mean_speed_kmh:.2f} km/h, "
                f"mean step {row.mean_step_km:.2f} km, radius {row.mean_radius_of_gyration_km:.2f} km."
            )
    lines.append("")
    return "\n".join(lines)


def run_sf_scenario(
    args: argparse.Namespace,
    *,
    tolerance_km: float,
    sample_interval: int,
    scenario_dir: Path,
    samples_df: pd.DataFrame,
    dataset_metadata: dict[str, object],
) -> None:
    """Run baselines + VQ for one (grid precision × resampling interval); write CSVs and logs under scenario_dir."""
    args = copy.copy(args)
    args.tolerance_km = float(tolerance_km)
    args.tolerance_kms = [float(tolerance_km)]
    scenario_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"[experiment] scenario tolerance_km={tolerance_km} sample_interval_min={sample_interval} dir={scenario_dir.name}",
        flush=True,
    )

    write_json(
        scenario_dir / "scenario_metadata.json",
        {
            "tolerance_km": tolerance_km,
            "sample_interval_min": sample_interval,
            "vq_d_target_scaled": vq_reconstruction_target_scaled(tolerance_km, args.coordinate_scale_km),
        },
    )

    _init_scenario_training_losses(scenario_dir, tolerance_km, sample_interval, args)
    wrote_vq_loss_section = False

    segment_results: list[dict[str, object]] = []
    latent_occurrences: list[pd.DataFrame] = []

    grid_cell_km = tolerance_km

    for interpolation_method in args.interpolation_methods:
        subset = samples_df.loc[
            (samples_df["sample_interval_min"] == sample_interval)
            & (samples_df["interpolation_method"] == interpolation_method)
        ].copy()
        if subset.empty:
            continue

        segment_arrays = extract_segment_arrays(subset)
        if args.max_segments is not None:
            segment_arrays = segment_arrays[: args.max_segments]
        all_xy = np.concatenate([segment["xy_km"] for segment in segment_arrays], axis=0)
        center_km = all_xy.mean(axis=0)

        for segment in segment_arrays:
            segment_results.extend(
                evaluate_baselines(
                    segment=segment,
                    cell_size_km=grid_cell_km,
                    markov_alpha=args.markov_alpha,
                    tolerance_km=tolerance_km,
                )
            )

        train_windows = build_training_windows(
            segment_arrays=segment_arrays,
            time_steps=args.time_steps,
            stride=args.window_stride,
            center_km=center_km,
            scale_km=args.coordinate_scale_km,
            max_segments=args.max_segments,
        )
        if len(train_windows) == 0 or args.skip_vq:
            continue

        d_target_scaled = vq_reconstruction_target_scaled(grid_cell_km, args.coordinate_scale_km)
        for codebook_size in args.codebook_sizes:
            log_path = scenario_dir / f"train_interval{sample_interval}_{interpolation_method}_K{codebook_size}.log"
            model = train_vqvae(
                train_windows=train_windows,
                args=args,
                codebook_size=codebook_size,
                d_target_scaled=d_target_scaled,
                log_path=log_path,
            )
            _append_scenario_training_loss_section(
                scenario_dir,
                sample_interval=sample_interval,
                interpolation_method=str(interpolation_method),
                codebook_size=codebook_size,
                log_path=log_path,
            )
            wrote_vq_loss_section = True
            save_vqvae_checkpoint(
                model,
                scenario_dir,
                tolerance_km=tolerance_km,
                sample_interval=sample_interval,
                interpolation_method=str(interpolation_method),
                codebook_size=codebook_size,
                d_target_scaled=d_target_scaled,
                center_km=center_km,
                args=args,
            )
            for segment in segment_arrays:
                vq_row, occurrence_df = evaluate_vqvae(
                    segment=segment,
                    model=model,
                    center_km=center_km,
                    scale_km=args.coordinate_scale_km,
                    args=args,
                    codebook_size=codebook_size,
                )
                segment_results.append(vq_row)
                if not occurrence_df.empty:
                    latent_occurrences.append(occurrence_df)

    if not args.skip_vq and not wrote_vq_loss_section:
        with (scenario_dir / "scenario_training_losses.txt").open("a", encoding="utf-8") as handle:
            handle.write("\n# No VQ training blocks: no training windows or empty segments for this scenario.\n")

    results_df = pd.DataFrame(segment_results)
    results_df.to_csv(scenario_dir / "segment_level_results.csv", index=False)

    group_extra = ["tolerance_km"]
    if results_df.empty:
        user_df = pd.DataFrame()
        trajectory_df = pd.DataFrame()
        label_df = pd.DataFrame()
    else:
        user_df = aggregate_weighted(
            df=results_df.dropna(subset=["predictability"]),
            group_cols=["user_id", "method", "sample_interval_min", "interpolation_method", "codebook_size", *group_extra],
            value_cols=["predictability", "entropy_rate_bits", "actual_distortion_km", "reconstruction_rmse_km"],
        )
        trajectory_df = aggregate_weighted(
            df=results_df.dropna(subset=["predictability"]),
            group_cols=["trajectory_id", "method", "sample_interval_min", "interpolation_method", "codebook_size", *group_extra],
            value_cols=["predictability", "entropy_rate_bits", "actual_distortion_km", "reconstruction_rmse_km"],
        )
        label_df = aggregate_weighted(
            df=results_df.dropna(subset=["predictability"]),
            group_cols=["dominant_mode", "method", "sample_interval_min", "interpolation_method", "codebook_size", *group_extra],
            value_cols=["predictability", "entropy_rate_bits", "actual_distortion_km", "reconstruction_rmse_km"],
        )
    user_df.to_csv(scenario_dir / "user_level_results.csv", index=False)
    save_user_predictability_exports(user_df, scenario_dir)
    trajectory_df.to_csv(scenario_dir / "trajectory_level_results.csv", index=False)
    label_df.to_csv(scenario_dir / "label_level_results.csv", index=False)

    occurrences_df = pd.concat(latent_occurrences, ignore_index=True) if latent_occurrences else pd.DataFrame()
    if occurrences_df.empty:
        occurrences_df = pd.DataFrame(
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
                "tolerance_km",
                "raw_code",
                "reindexed_code",
                "recon_x_km",
                "recon_y_km",
                "step_km",
                "speed_kmh",
                "radius_of_gyration_km",
                "codebook_size",
            ]
        )
    occurrences_df.to_csv(scenario_dir / "latent_code_occurrences.csv", index=False)
    code_summary_df = summarize_latent_codes(occurrences_df)
    code_summary_df.to_csv(scenario_dir / "latent_code_summary.csv", index=False)

    save_summary_plots(results_df, scenario_dir)

    reviewer_summary = build_reviewer_summary(
        results_df=results_df,
        code_summary_df=code_summary_df,
        metadata=dataset_metadata,
        codebook_sizes=args.codebook_sizes,
    )
    (scenario_dir / "reviewer_summary.md").write_text(reviewer_summary, encoding="utf-8")
    write_json(scenario_dir / "experiment_args.json", vars(args))


def main() -> None:
    args = parse_args()
    if args.tolerance_km is not None:
        args.tolerance_kms = [float(args.tolerance_km)]
    _apply_hpc_scenario_task(args)
    _apply_hpc_task_slices(args)
    if torch is None:
        args.skip_vq = True
    output_dir = resolve_output_dir(args.output_dir, args.run_name)
    dataset_path = output_dir / "sf_samples.pkl"
    run_metadata_path = output_dir / "run_metadata.json"

    print(
        "[experiment] dataset=sf output_dir=" + str(output_dir),
        flush=True,
    )
    print(
        f"[experiment] tolerance_kms={args.tolerance_kms} sample_intervals={args.sample_intervals} "
        f"codebook_sizes={args.codebook_sizes}",
        flush=True,
    )

    samples_df = build_sf_resampled_dataset(
        data_path=args.data_dir,
        sample_intervals=args.sample_intervals,
        interpolation_methods=args.interpolation_methods,
        require_mode=True,
        min_points=args.min_points,
        max_users=args.max_users,
        selected_user_ids=None,
        projection=args.projection,
    )
    n_seg = int(samples_df["segment_id"].nunique()) if not samples_df.empty else 0
    print(f"[experiment] resampled samples: rows={len(samples_df)} segments={n_seg}", flush=True)
    samples_df.to_pickle(dataset_path)
    save_user_label_exports(samples_df, output_dir)

    metadata = compute_dataset_metadata(samples_df)
    metadata.update(
        {
            "dataset": "sf",
            "dataset_label": "Cabspotting / San Francisco",
            "data_dir": args.data_dir,
            "sample_intervals": args.sample_intervals,
            "interpolation_methods": args.interpolation_methods,
            "codebook_sizes": args.codebook_sizes,
            "tolerance_kms": args.tolerance_kms,
            "coordinate_scale_km": args.coordinate_scale_km,
            "projection": args.projection,
            "skip_vq": args.skip_vq,
            "torch_available": torch is not None,
            "hpc_coarse_task": args.hpc_coarse_task,
            "hpc_fine_task": args.hpc_fine_task,
            "hpc_scenario_task": args.hpc_scenario_task,
            "run_name": args.run_name,
            "user_label_export_files": [
                "user_resampled_label_mode_counts.csv",
                "user_resampled_label_mode_counts_wide.csv",
                "user_label_mode_fraction_by_user.csv",
                "user_original_label_intervals.csv",
            ],
        }
    )
    write_json(run_metadata_path, metadata)

    for tol_km in args.tolerance_kms:
        for sample_interval in args.sample_intervals:
            scenario_dir = output_dir / scenario_subdir_name(tol_km, sample_interval)
            run_sf_scenario(
                args,
                tolerance_km=tol_km,
                sample_interval=sample_interval,
                scenario_dir=scenario_dir,
                samples_df=samples_df,
                dataset_metadata=metadata,
            )

    concat_all_scenarios_user_predictability(output_dir, args)
    write_json(output_dir / "experiment_args.json", vars(args))


if __name__ == "__main__":
    main()
