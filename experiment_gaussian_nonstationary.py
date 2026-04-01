from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from ctw_estimate import CTWEntropy
from model import EC_VQVAE, model_train, return_zq_list
from sequence_gen import (
    find_p,
    generate_piecewise_gaussian_markov_sequence,
    sliding_window_batches,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Non-stationary Gaussian Markov stress test for predictability robustness."
    )
    parser.add_argument("--output-dir", type=Path, default=Path("gaussian_nonstationary_results"))
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-length", type=int, default=120_000)
    parser.add_argument("--val-length", type=int, default=60_000)
    parser.add_argument("--dimension", type=int, default=2)
    parser.add_argument("--ar-order", type=int, default=5)
    parser.add_argument("--switch-frac", type=float, default=0.5)
    parser.add_argument(
        "--shift-levels",
        nargs="+",
        type=float,
        default=[0.0, 0.33, 0.66, 1.0],
        help="Non-stationarity strength levels. 0 = stationary reference, 1 = full regime shift.",
    )
    parser.add_argument(
        "--scenario",
        choices=["matrix_shift", "noise_shift", "mean_shift"],
        default="matrix_shift",
        help="How non-stationarity is introduced at the regime switch.",
    )
    parser.add_argument("--noise-std-pre", type=float, default=1.0)
    parser.add_argument("--noise-std-post", type=float, default=2.0)
    parser.add_argument("--mean-shift-scale", type=float, default=1.5)
    parser.add_argument("--max-radius-pre", type=float, default=0.25)
    parser.add_argument("--max-radius-post", type=float, default=0.55)
    parser.add_argument("--time-steps", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--num-epochs", type=int, default=20)
    parser.add_argument("--pretrain-epochs", type=int, default=1)
    parser.add_argument("--hidden-channels", type=int, default=32)
    parser.add_argument("--embedding-dim", type=int, default=6)
    parser.add_argument("--num-conv-layers", type=int, default=3)
    parser.add_argument("--kernel-size", type=int, default=13)
    parser.add_argument("--codebook-size", type=int, default=128)
    parser.add_argument("--commitment-cost", type=float, default=0.25)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=0.2)
    parser.add_argument("--step-size", type=int, default=5)
    parser.add_argument(
        "--distortions",
        nargs="+",
        type=float,
        default=[1.50, 1.00, 0.75, 0.50],
        help="D_target values for the ALM reconstruction constraint.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return parser.parse_args()


def make_output_dir(base_dir: Path, run_name: str | None) -> Path:
    if run_name:
        safe_name = run_name.strip().replace("..", "_")
        run_dir = base_dir / safe_name
    else:
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = base_dir / f"gaussian_nonstationary_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def write_json(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def companion_spectral_radius(a_matrices: list[np.ndarray]) -> float:
    order = len(a_matrices)
    dimension = a_matrices[0].shape[0]
    companion = np.zeros((order * dimension, order * dimension), dtype=np.complex128)
    companion[:dimension, :] = np.hstack(a_matrices)
    for idx in range(1, order):
        companion[idx * dimension : (idx + 1) * dimension, (idx - 1) * dimension : idx * dimension] = np.eye(
            dimension
        )
    eigvals = np.linalg.eigvals(companion)
    return float(np.max(np.abs(eigvals)))


def interpolate_post_matrices(
    a_pre: list[np.ndarray],
    a_post_full: list[np.ndarray],
    level: float,
) -> list[np.ndarray]:
    if level <= 0.0:
        return [mat.copy() for mat in a_pre]
    if level >= 1.0:
        candidate = [mat.copy() for mat in a_post_full]
    else:
        candidate = [pre + level * (post - pre) for pre, post in zip(a_pre, a_post_full)]

    radius = companion_spectral_radius(candidate)
    if radius < 1.0 - 1e-8:
        return candidate

    shrink = min(0.98 / max(radius, 1e-12), 1.0)
    stabilized = [shrink * mat for mat in candidate]
    return stabilized


def entropy_predictability(codes: list[int]) -> tuple[float, float, int]:
    if not codes:
        return float("nan"), float("nan"), 0
    max_symbol = max(codes) + 1
    if max_symbol <= 1:
        return 0.0, 1.0, 1
    entropy_rate = float(CTWEntropy(max_symbol=max_symbol).calculate_entropy_rate(codes))
    predictability = float(find_p(entropy_rate, max_symbol))
    return entropy_rate, predictability, max_symbol


def evaluate_sequence(
    model: EC_VQVAE,
    x: np.ndarray,
    *,
    device: str,
    time_steps: int,
    kernel_size: int,
    batch_size: int,
) -> dict[str, float]:
    q_sequence, recon_rmse = return_zq_list(
        X=torch.tensor(x, dtype=torch.float32),
        model=model,
        device=torch.device(device),
        time_steps=time_steps,
        kernel_size=kernel_size,
        batch_size=batch_size,
    )
    entropy_rate, predictability, alphabet_size = entropy_predictability(q_sequence)
    return {
        "recon_rmse": float(recon_rmse),
        "entropy_rate_bits": float(entropy_rate),
        "predictability": float(predictability),
        "alphabet_size": int(alphabet_size),
        "num_codes": int(len(q_sequence)),
    }


def main() -> None:
    args = parse_args()
    output_dir = make_output_dir(args.output_dir, args.run_name)
    write_json(output_dir / "experiment_args.json", vars(args))

    rng = np.random.default_rng(args.seed)

    # Sample a single base generator family so that stationary and non-stationary runs differ only by controlled shift level.
    _, a_pre, a_post_full, sampled_mean_shift = generate_piecewise_gaussian_markov_sequence(
        N=max(args.ar_order + 2, 16),
        D=args.dimension,
        R=args.ar_order,
        rng=rng,
        switch_frac=args.switch_frac,
        scenario=args.scenario,
        noise_std_pre=args.noise_std_pre,
        noise_std_post=args.noise_std_post,
        mean_shift_scale=args.mean_shift_scale,
        max_radius_pre=args.max_radius_pre,
        max_radius_post=args.max_radius_post,
    )

    rows: list[dict[str, float | int | str]] = []
    loss_lines: list[str] = []

    for level in args.shift_levels:
        if args.scenario == "matrix_shift":
            a_post_level = interpolate_post_matrices(a_pre, a_post_full, float(level))
            noise_std_post_level = args.noise_std_pre
            mean_shift_vector = np.zeros(args.dimension, dtype=float)
        elif args.scenario == "noise_shift":
            a_post_level = [mat.copy() for mat in a_pre]
            noise_std_post_level = args.noise_std_pre + float(level) * (args.noise_std_post - args.noise_std_pre)
            mean_shift_vector = np.zeros(args.dimension, dtype=float)
        else:
            a_post_level = [mat.copy() for mat in a_pre]
            noise_std_post_level = args.noise_std_pre
            mean_shift_vector = float(level) * np.asarray(sampled_mean_shift, dtype=float)

        level_seed = args.seed + int(round(float(level) * 1000)) + 17
        level_rng = np.random.default_rng(level_seed)
        train_x, _, _, _ = generate_piecewise_gaussian_markov_sequence(
            N=args.train_length,
            D=args.dimension,
            R=args.ar_order,
            rng=level_rng,
            switch_frac=args.switch_frac,
            scenario=args.scenario,
            noise_std_pre=args.noise_std_pre,
            noise_std_post=noise_std_post_level,
            mean_shift_scale=args.mean_shift_scale,
            max_radius_pre=args.max_radius_pre,
            max_radius_post=args.max_radius_post,
            A_matrices_pre=a_pre,
            A_matrices_post=a_post_level,
            mean_shift_vector=mean_shift_vector,
        )
        val_x, _, _, _ = generate_piecewise_gaussian_markov_sequence(
            N=args.val_length,
            D=args.dimension,
            R=args.ar_order,
            rng=level_rng,
            switch_frac=args.switch_frac,
            scenario=args.scenario,
            noise_std_pre=args.noise_std_pre,
            noise_std_post=noise_std_post_level,
            mean_shift_scale=args.mean_shift_scale,
            max_radius_pre=args.max_radius_pre,
            max_radius_post=args.max_radius_post,
            A_matrices_pre=a_pre,
            A_matrices_post=a_post_level,
            mean_shift_vector=mean_shift_vector,
        )

        np.save(output_dir / f"X_train_level_{float(level):.2f}.npy", train_x)
        np.save(output_dir / f"X_val_level_{float(level):.2f}.npy", val_x)

        windows = sliding_window_batches(train_x, args.time_steps, stride=1)
        train_loader = DataLoader(windows, batch_size=args.batch_size, shuffle=True, drop_last=True)

        split_index = int(round(args.val_length * args.switch_frac))
        first_half = val_x[:split_index]
        second_half = val_x[split_index:]

        for distortion in args.distortions:
            model = EC_VQVAE(
                in_channels=args.dimension,
                hidden_channels=args.hidden_channels,
                codebook_size=args.codebook_size,
                embedding_dim=args.embedding_dim,
                commitment_cost=args.commitment_cost,
                time_steps=args.time_steps,
                num_conv_layers=args.num_conv_layers,
                kernel_size=args.kernel_size,
            ).to(args.device)

            train_log_path = output_dir / f"train_level{float(level):.2f}_D{distortion:.2f}.log"
            model_train(
                model=model,
                dataloader=train_loader,
                device=torch.device(args.device),
                num_epochs=args.num_epochs,
                output_file=str(train_log_path),
                pretrain_epochs=args.pretrain_epochs,
                step_size=args.step_size,
                sigma=args.sigma,
                gamma=args.gamma,
                D_target=float(distortion),
                lr=args.lr,
            )

            full_metrics = evaluate_sequence(
                model,
                val_x,
                device=args.device,
                time_steps=args.time_steps,
                kernel_size=args.kernel_size,
                batch_size=args.eval_batch_size,
            )
            first_metrics = evaluate_sequence(
                model,
                first_half,
                device=args.device,
                time_steps=args.time_steps,
                kernel_size=args.kernel_size,
                batch_size=args.eval_batch_size,
            )
            second_metrics = evaluate_sequence(
                model,
                second_half,
                device=args.device,
                time_steps=args.time_steps,
                kernel_size=args.kernel_size,
                batch_size=args.eval_batch_size,
            )

            weighted_halves_avg = 0.5 * (first_metrics["predictability"] + second_metrics["predictability"])
            row = {
                "scenario": args.scenario,
                "shift_level": float(level),
                "distortion_target": float(distortion),
                "switch_frac": float(args.switch_frac),
                "full_predictability": full_metrics["predictability"],
                "first_half_predictability": first_metrics["predictability"],
                "second_half_predictability": second_metrics["predictability"],
                "predictability_gap_halves": abs(first_metrics["predictability"] - second_metrics["predictability"]),
                "full_vs_halves_deviation": abs(full_metrics["predictability"] - weighted_halves_avg),
                "full_entropy_rate_bits": full_metrics["entropy_rate_bits"],
                "first_half_entropy_rate_bits": first_metrics["entropy_rate_bits"],
                "second_half_entropy_rate_bits": second_metrics["entropy_rate_bits"],
                "full_recon_rmse": full_metrics["recon_rmse"],
                "first_half_recon_rmse": first_metrics["recon_rmse"],
                "second_half_recon_rmse": second_metrics["recon_rmse"],
                "full_alphabet_size": full_metrics["alphabet_size"],
                "first_half_alphabet_size": first_metrics["alphabet_size"],
                "second_half_alphabet_size": second_metrics["alphabet_size"],
                "noise_std_post_level": float(noise_std_post_level),
                "mean_shift_norm": float(np.linalg.norm(mean_shift_vector)),
                "post_spectral_radius": companion_spectral_radius(a_post_level),
            }
            rows.append(row)
            loss_lines.append(
                " | ".join(
                    [
                        f"level={float(level):.2f}",
                        f"D_target={distortion:.2f}",
                        f"full_pi={row['full_predictability']:.4f}",
                        f"first_pi={row['first_half_predictability']:.4f}",
                        f"second_pi={row['second_half_predictability']:.4f}",
                        f"gap={row['predictability_gap_halves']:.4f}",
                        f"dev={row['full_vs_halves_deviation']:.4f}",
                    ]
                )
            )

    results_df = pd.DataFrame(rows).sort_values("distortion_target", ascending=False)
    results_df.to_csv(output_dir / "nonstationary_gaussian_summary.csv", index=False)
    level_summary_df = (
        results_df.groupby("shift_level", as_index=False)[
            [
                "full_predictability",
                "predictability_gap_halves",
                "full_vs_halves_deviation",
                "full_entropy_rate_bits",
                "full_recon_rmse",
            ]
        ]
        .mean()
        .sort_values("shift_level")
    )
    level_summary_df.to_csv(output_dir / "nonstationary_gaussian_level_summary.csv", index=False)
    (output_dir / "scenario_training_losses.txt").write_text("\n".join(loss_lines) + "\n", encoding="utf-8")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for level, group in results_df.groupby("shift_level", sort=True):
        ordered = group.sort_values("full_recon_rmse")
        axes[0].plot(
            ordered["full_recon_rmse"],
            ordered["full_entropy_rate_bits"],
            marker="o",
            label=f"Level {level:.2f}",
        )
        axes[1].plot(
            ordered["full_recon_rmse"],
            ordered["full_predictability"],
            marker="^",
            label=f"Level {level:.2f}",
        )

    axes[0].set_xlabel("Distortion D (reconstruction RMSE)")
    axes[0].set_ylabel("Rate R (bits)")
    axes[0].set_title("(a) Rate-Distortion under non-stationarity")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].set_xlabel("Distortion D (reconstruction RMSE)")
    axes[1].set_ylabel("Predictability")
    axes[1].set_title("(b) Predictability-Distortion under non-stationarity")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_dir / "distortion_rate_curve_nonstationary.png", dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for distortion, group in results_df.groupby("distortion_target", sort=True):
        ordered = group.sort_values("shift_level")
        axes[0].plot(
            ordered["shift_level"],
            ordered["predictability_gap_halves"],
            marker="o",
            label=f"D={distortion:.2f}",
        )
        axes[1].plot(
            ordered["shift_level"],
            ordered["full_vs_halves_deviation"],
            marker="s",
            label=f"D={distortion:.2f}",
        )

    axes[0].set_xlabel("Non-stationarity level")
    axes[0].set_ylabel("Predictability gap between halves")
    axes[0].set_title("(a) Half-to-half robustness gap")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].set_xlabel("Non-stationarity level")
    axes[1].set_ylabel("|Full estimate - average half estimate|")
    axes[1].set_title("(b) Full-vs-local deviation")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_dir / "robustness_vs_nonstationarity.png", dpi=200)
    plt.close(fig)

    metadata = {
        "scenario": args.scenario,
        "seed": int(args.seed),
        "train_length": int(args.train_length),
        "val_length": int(args.val_length),
        "dimension": int(args.dimension),
        "ar_order": int(args.ar_order),
        "switch_frac": float(args.switch_frac),
        "noise_std_pre": float(args.noise_std_pre),
        "noise_std_post": float(args.noise_std_post),
        "shift_levels": [float(item) for item in args.shift_levels],
        "sampled_mean_shift": np.asarray(sampled_mean_shift, dtype=float).tolist(),
        "a_pre": [mat.tolist() for mat in a_pre],
        "a_post_full": [mat.tolist() for mat in a_post_full],
        "distortions": [float(item) for item in args.distortions],
        "device": str(args.device),
    }
    write_json(output_dir / "scenario_metadata.json", metadata)


if __name__ == "__main__":
    main()
