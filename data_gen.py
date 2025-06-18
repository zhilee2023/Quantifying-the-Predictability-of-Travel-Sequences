import os
import datetime

import numpy as np
import matplotlib.pyplot as plt
from sequence_gen import generate_gaussian_markov_sequence, compute_rate_distortion_vector

# -----------------------------------
# Configuration
# -----------------------------------
SEQUENCE_LEN = 1_000_000    # Number of samples in training split
DIMENSION = 2               # Dimensionality of the AR process
AR_ORDER = 5                # Order of the autoregressive model
SIGMA_Z = 1.0               # Standard deviation of white noise
THETA_VALUES = np.logspace(-3, 2, 50)  # Parameter for rate-distortion computation
N_FREQ = 100_000            # Frequency resolution for theoretical RD

# Derived length (we generate a bit more to account for initial AR order)
TOTAL_LENGTH = SEQUENCE_LEN * 3 + AR_ORDER

# Output paths with timestamp
TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = "data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Filenames
X_TRAIN_PATH = os.path.join(OUTPUT_DIR, "X_train.npy")
X_VAL_PATH = os.path.join(OUTPUT_DIR, "X_val.npy")
COEFF_PATH = os.path.join(OUTPUT_DIR, "coeff.npy")
D_VALS_PATH = os.path.join(OUTPUT_DIR, "D_vals.npy")
R_VALS_PATH = os.path.join(OUTPUT_DIR, "R_vals.npy")
PLOT_PATH = os.path.join(OUTPUT_DIR, "distortion_rate_curve.png")

# -----------------------------------
# Generate Gaussian Markov Sequence
# -----------------------------------

X_full, A_matrices = generate_gaussian_markov_sequence(
    N=TOTAL_LENGTH,
    D=DIMENSION,
    R=AR_ORDER,
    noise_std=SIGMA_Z,
)

# Split into training and validation
X_train = X_full[:SEQUENCE_LEN]
X_val = X_full[SEQUENCE_LEN : SEQUENCE_LEN * 2]  # next SEQUENCE_LEN samples for validation

# -----------------------------------
# Theoretical Rate-Distortion Computation
# -----------------------------------
D_vals, R_vals = compute_rate_distortion_vector(
    A_matrices,
    sigma_z=SIGMA_Z,
    theta_vals=THETA_VALUES,
    n_freq=N_FREQ,
)

# -----------------------------------
# Save Data and Results
# -----------------------------------
np.save(X_TRAIN_PATH, X_train)
np.save(X_VAL_PATH, X_val)
np.save(COEFF_PATH, np.array(A_matrices))
np.save(D_VALS_PATH, D_vals)
np.save(R_VALS_PATH, R_vals)


# -----------------------------------
# Plot Distortion-Rate Curve
# -----------------------------------
plt.figure(figsize=(8, 6))
plt.plot(R_vals, D_vals, linewidth=2, label="Theoretical RD")
plt.xlabel("Rate R (bits)")
plt.ylabel("Distortion D")
plt.title("Rate-Distortion Curve for AR(2) Gaussian Process")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(PLOT_PATH)
plt.close()

