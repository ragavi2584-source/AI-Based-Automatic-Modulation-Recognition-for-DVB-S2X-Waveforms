"""
generate_dataset.py
AI-Based Automatic Modulation Recognition for DVB-S2X

Generates synthetic I/Q samples for:
1. 16-APSK
2. 32-APSK
3. 64-APSK

Author: Hackathon Project
"""

import os
import numpy as np
from tqdm import tqdm

# -----------------------------
# Configuration
# -----------------------------
NUM_SAMPLES = 5000          # Samples per modulation
IQ_LENGTH = 128             # I/Q points per sample
SNR_LEVELS = [-10, -5, 0, 5, 10, 15, 20]

OUTPUT_DIR = "dataset"

MODULATIONS = {
    0: "16APSK",
    1: "32APSK",
    2: "64APSK"
}

np.random.seed(42)

# -----------------------------
# Create Output Folder
# -----------------------------
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)


# -----------------------------
# Add AWGN Noise
# -----------------------------
def add_awgn(signal, snr_db):

    signal_power = np.mean(np.abs(signal) ** 2)

    snr_linear = 10 ** (snr_db / 10)

    noise_power = signal_power / snr_linear

    noise = np.sqrt(noise_power / 2) * (
        np.random.randn(*signal.shape)
        + 1j * np.random.randn(*signal.shape)
    )

    return signal + noise


# -----------------------------
# Normalize Signal
# -----------------------------
def normalize(signal):

    signal = signal / np.sqrt(np.mean(np.abs(signal) ** 2))

    return signal


# -----------------------------
# Save Dataset
# -----------------------------
def save_dataset(X, y):

    np.save(os.path.join(OUTPUT_DIR, "train_x.npy"), X)
    np.save(os.path.join(OUTPUT_DIR, "train_y.npy"), y)

    print("Dataset saved successfully.")
    print("Shape:", X.shape)