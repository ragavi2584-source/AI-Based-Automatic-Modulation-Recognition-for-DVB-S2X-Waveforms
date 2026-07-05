"""
visualization.py
Visualization Module for Automatic Modulation Recognition
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from scipy.fft import fft, fftfreq
from sklearn.metrics import confusion_matrix
import itertools

OUTPUT_DIR = "static/plots"

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)


# ------------------------------------------
# Constellation Diagram
# ------------------------------------------

def plot_constellation(signal,
                       filename="constellation.png"):

    plt.figure(figsize=(6,6))

    plt.scatter(
        np.real(signal),
        np.imag(signal),
        s=10,
        alpha=0.7
    )

    plt.title("Constellation Diagram")
    plt.xlabel("In-phase (I)")
    plt.ylabel("Quadrature (Q)")
    plt.grid(True)

    path = os.path.join(OUTPUT_DIR, filename)

    plt.savefig(path)

    plt.close()

    return path


# ------------------------------------------
# Time Domain Plot
# ------------------------------------------

def plot_waveform(signal,
                  filename="waveform.png"):

    plt.figure(figsize=(10,4))

    plt.plot(
        np.real(signal),
        label="I Channel"
    )

    plt.plot(
        np.imag(signal),
        label="Q Channel"
    )

    plt.title("Time Domain Signal")

    plt.xlabel("Sample")

    plt.ylabel("Amplitude")

    plt.legend()

    plt.grid(True)

    path = os.path.join(OUTPUT_DIR, filename)

    plt.savefig(path)

    plt.close()

    return path


# ------------------------------------------
# Frequency Spectrum
# ------------------------------------------

def plot_fft(signal,
             filename="fft.png"):

    N = len(signal)

    yf = fft(signal)

    xf = fftfreq(N)

    plt.figure(figsize=(10,4))

    plt.plot(
        xf,
        np.abs(yf)
    )

    plt.title("Frequency Spectrum")

    plt.xlabel("Normalized Frequency")

    plt.ylabel("Magnitude")

    plt.grid(True)

    path = os.path.join(OUTPUT_DIR, filename)

    plt.savefig(path)

    plt.close()

    return path
# ------------------------------------------
# Spectrogram
# ------------------------------------------

def plot_spectrogram(signal,
                     filename="spectrogram.png"):

    plt.figure(figsize=(10,4))

    plt.specgram(
        np.real(signal),
        NFFT=32,
        Fs=1,
        noverlap=16
    )

    plt.title("Signal Spectrogram")

    plt.xlabel("Time")

    plt.ylabel("Frequency")

    plt.colorbar(label="Power")

    path = os.path.join(OUTPUT_DIR, filename)

    plt.savefig(path)

    plt.close()

    return path


# ------------------------------------------
# Prediction Confidence
# ------------------------------------------

def plot_confidence(confidence_vector,
                    class_names,
                    filename="confidence.png"):

    plt.figure(figsize=(6,4))

    plt.bar(
        class_names,
        confidence_vector
    )

    plt.ylim(0,1)

    plt.ylabel("Confidence")

    plt.title("Prediction Confidence")

    plt.grid(axis="y")

    path = os.path.join(OUTPUT_DIR, filename)

    plt.savefig(path)

    plt.close()

    return path


# ------------------------------------------
# Confusion Matrix
# ------------------------------------------

def plot_confusion_matrix(
        y_true,
        y_pred,
        class_names,
        filename="confusion_matrix.png"):

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(6,6))

    plt.imshow(
        cm,
        interpolation="nearest",
        cmap=plt.cm.Blues
    )

    plt.title("Confusion Matrix")

    plt.colorbar()

    tick_marks = np.arange(len(class_names))

    plt.xticks(
        tick_marks,
        class_names,
        rotation=45
    )

    plt.yticks(
        tick_marks,
        class_names
    )

    thresh = cm.max() / 2

    for i, j in itertools.product(
            range(cm.shape[0]),
            range(cm.shape[1])):

        plt.text(
            j,
            i,
            format(cm[i, j], "d"),
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black"
        )

    plt.ylabel("True Label")

    plt.xlabel("Predicted Label")

    plt.tight_layout()

    path = os.path.join(OUTPUT_DIR, filename)

    plt.savefig(path)

    plt.close()

    return path