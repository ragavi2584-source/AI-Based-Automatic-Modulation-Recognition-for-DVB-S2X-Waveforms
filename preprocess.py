"""
preprocess.py
Preprocessing utilities for AI-Based Automatic Modulation Recognition
"""

import numpy as np
from scipy.signal import butter, filtfilt

# -----------------------------
# Normalize IQ Samples
# -----------------------------
def normalize_iq(signal):
    """
    Normalize signal power to 1.
    """
    power = np.mean(np.abs(signal) ** 2)

    if power == 0:
        return signal

    return signal / np.sqrt(power)


# -----------------------------
# Remove DC Offset
# -----------------------------
def remove_dc(signal):
    """
    Removes DC component.
    """
    return signal - np.mean(signal)


# -----------------------------
# Low Pass Filter
# -----------------------------
def lowpass_filter(signal, cutoff=0.2, order=5):

    b, a = butter(order, cutoff)

    real = filtfilt(b, a, np.real(signal))
    imag = filtfilt(b, a, np.imag(signal))

    return real + 1j * imag


# -----------------------------
# Convert Complex to IQ Matrix
# -----------------------------
def complex_to_iq(signal):

    i = np.real(signal)

    q = np.imag(signal)

    return np.column_stack((i, q))


# -----------------------------
# Pad or Trim Signal
# -----------------------------
def resize_signal(signal, length=128):

    if len(signal) > length:
        signal = signal[:length]

    elif len(signal) < length:

        padding = np.zeros(length - len(signal), dtype=complex)

        signal = np.concatenate((signal, padding))

    return signal


# -----------------------------
# Complete Preprocessing
# -----------------------------
def preprocess_signal(signal):

    signal = remove_dc(signal)

    signal = normalize_iq(signal)

    signal = lowpass_filter(signal)

    signal = resize_signal(signal)

    signal = complex_to_iq(signal)

    return signal


# -----------------------------
# Batch Preprocessing
# -----------------------------
def preprocess_dataset(dataset):

    processed = []

    for signal in dataset:
        processed.append(preprocess_signal(signal))

    return np.array(processed)


# -----------------------------
# Load Dataset
# -----------------------------
def load_dataset(x_path, y_path):

    X = np.load(x_path)

    y = np.load(y_path)

    return X, y


# -----------------------------
# Save Dataset
# -----------------------------
def save_dataset(X, y, x_path, y_path):

    np.save(x_path, X)

    np.save(y_path, y)

    print("Preprocessed dataset saved.")


# -----------------------------
# Example Usage
# -----------------------------
if __name__ == "__main__":

    X, y = load_dataset(
        "dataset/train_x.npy",
        "dataset/train_y.npy"
    )

    X = preprocess_dataset(X)

    save_dataset(
        X,
        y,
        "dataset/train_x_processed.npy",
        "dataset/train_y_processed.npy"
    )

    print("Preprocessing completed successfully.")