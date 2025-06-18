import os
import datetime

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from sequence_gen import (
    sliding_window_batches,
    cal_entropy_rate,
    save_parameters_to_json,
    compute_distances,
)
from model import EC_VQVAE, model_train, return_zq_list

# -----------------------------------
# Configuration
# -----------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
T = 150  # time steps per sequence
BATCH_SIZE = 512
NUM_EPOCHS = 20
PRETRAIN_EPOCHS = 1

# Model hyperparameters
N_BLOCK = 16
BETA = 1.0
SIGMA = 1.0
KERNEL_SIZE = 13
HIDDEN_CHANNELS = 32
EMBEDDING_DIM = 6
NUM_CONV_LAYERS = 3
CODEBOOK_SIZE = 128

# Distortion thresholds to sweep (descending order)
DISTORTIONS = np.arange(1.75, 0.0, -0.1)


MACHINE_ID=os.environ["MACHINE_ID"]
# Paths
DATA_DIR = "data"
OUTPUT_BASE = "rate_distortion_results"
TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = f"{OUTPUT_BASE}_{MACHINE_ID}_{TIMESTAMP}"
os.makedirs(OUTPUT_DIR, exist_ok=True)
LOG_PATH = os.path.join(OUTPUT_DIR, "rate_distortion_results.txt")



# -----------------------------------
# Load data once
# -----------------------------------
X_train = np.load(os.path.join(DATA_DIR, "X_train.npy"))  # shape: (Total_T, feature_dim)
X_val = np.load(os.path.join(DATA_DIR, "X_val.npy"))
R_vals = np.load(os.path.join(DATA_DIR, "R_vals.npy"))
D_vals = np.load(os.path.join(DATA_DIR, "D_vals.npy"))

# Build DataLoader for training
# sliding_window_batches returns shape (num_samples, T, feature_dim)
train_dataset = sliding_window_batches(X_train, T, stride=1)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

# -----------------------------------
# Prepare storage for results
# -----------------------------------
rate_list = []
dist_list = []
predict_list = []
predict_base_list = []
dist_threshhold_list=[]
symbol_num=[]
# Write header to log
with open(LOG_PATH, "w") as log_file:
    log_file.write("Rate-Distortion Sweep\n")
    log_file.write(f"Timestamp: {TIMESTAMP}\n\n")

# -----------------------------------
# Main loop: sweep over distortion thresholds
# -----------------------------------
for dist_threshold in DISTORTIONS:
    # Instantiate a new model for each distortion target
    model = EC_VQVAE(
        in_channels=X_train.shape[1],
        hidden_channels=HIDDEN_CHANNELS,
        codebook_size=CODEBOOK_SIZE,
        embedding_dim=EMBEDDING_DIM,
        commitment_cost=0.25,
        time_steps=T,
        num_conv_layers=NUM_CONV_LAYERS,
        kernel_size=KERNEL_SIZE,
    ).to(DEVICE)

    # Log model size
    total_params = sum(p.numel() for p in model.parameters())
    with open(LOG_PATH, "a") as log_file:
        log_file.write(f"Distortion target: {dist_threshold:.2f}\n")
        #log_file.write(f"Total parameters: {total_params:,}\n")

    # Train the model
    model_train(
        model=model,
        dataloader=train_loader,
        device=DEVICE,
        num_epochs=NUM_EPOCHS,
        output_file=LOG_PATH,
        pretrain_epochs=PRETRAIN_EPOCHS,
        step_size=5,
        sigma=SIGMA,
        gamma=0.2,
        D_target=dist_threshold,
        lr=1e-3,
    )

    # Evaluate on validation split
    model.eval()
    q_sequence, avg_recon = return_zq_list(
        X=torch.tensor(X_val, dtype=torch.float32),
        model=model,
        device=DEVICE,
        time_steps=T,
        kernel_size=KERNEL_SIZE,
        batch_size=BATCH_SIZE,
    )

    # Compute rate, base rate, predictability metrics
    H, R_base, pi, Pi = cal_entropy_rate(q_sequence, D_vals, R_vals,dist_threshold)

    # Store results
    dist_threshhold_list.append(dist_threshold)
    rate_list.append(H)
    dist_list.append(avg_recon)
    predict_list.append(pi)
    predict_base_list.append(Pi)
    symbol_num.append(max(q_sequence)+1)
    # Compute “distances” or any additional metric
    distances = compute_distances(R_vals, D_vals, rate_list, dist_list)

    # Log metrics
    with open(LOG_PATH, "a") as log_file:
        log_file.write(
            f"  Rate     = {H:.4f}  |  Base Rate = {R_base:.4f}\n"
            f"  Recon    = {avg_recon:.4f}  |  Distances = {distances}\n"
            f"  Predict  = {pi:.4f}  |  Base Pred  = {Pi:.4f}\n"
            f"  Max Symbol = {max(q_sequence)}\n\n"
        )

# -----------------------------------
# After sweep: plot and save data
# -----------------------------------
pi_vals = np.array(predict_list)
Pi_vals = np.array(predict_base_list)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# (a) Rate-Distortion Curve
axes[0].plot(D_vals, R_vals, label="Baseline RD", linewidth=2)
axes[0].scatter(dist_list, rate_list, color="orange", marker="o", s=10, label="Sample RD")
axes[0].set_xlabel("Distortion D")
axes[0].set_ylabel("Rate R (bits)")
axes[0].set_title("(a) Rate-Distortion Comparison")
axes[0].grid(True)
axes[0].legend()

# (b) Predictability vs. Distortion
axes[1].scatter(dist_list, pi_vals, color="orange", marker="^", s=10, label="Sample")
axes[1].scatter(dist_list, Pi_vals, color="blue", marker="s", s=10, label="Baseline")
axes[1].set_xlabel("Distortion D")
axes[1].set_ylabel("Predictability")
axes[1].set_title("(b) Predictability-Distortion Comparison")
axes[1].grid(True)
axes[1].legend()

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "distortion_rate_curve.png"))
plt.close(fig)

# Save results to CSV
df_results = pd.DataFrame({
    "symbol_num":symbol_num,
    "dist_thresh":dist_threshhold_list,
    "Distortion_D": dist_list,
    "Rate_R_bits": rate_list,
    "Sample_Predictability_pi": pi_vals,
    "Baseline_Predictability_Pi": Pi_vals,
})
df_results.to_csv(os.path.join(OUTPUT_DIR, "distortion_rate_predictability_data.csv"), index=False)



# -----------------------------------
# Save experiment parameters
# -----------------------------------
params = {
    "T": T,
    "num_epochs": NUM_EPOCHS,
    "pretrain_epochs": PRETRAIN_EPOCHS,
    "beta": BETA,
    "sigma": SIGMA,
    "kernel_size": KERNEL_SIZE,
    "hidden_channels": HIDDEN_CHANNELS,
    "embedding_dim": EMBEDDING_DIM,
    "num_conv_layers": NUM_CONV_LAYERS,
    "codebook_size": CODEBOOK_SIZE,
    "distortions": DISTORTIONS.tolist(),
}
save_parameters_to_json(params, OUTPUT_DIR, TIMESTAMP)