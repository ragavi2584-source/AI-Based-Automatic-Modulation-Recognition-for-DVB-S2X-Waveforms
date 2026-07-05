"""
predict.py
AI-Based Automatic Modulation Recognition (AMR)
Prediction Module

Author : Your Team
"""

import os
import numpy as np
from tensorflow.keras.models import load_model

from preprocess import preprocess_signal

# -------------------------------------------------------
# Configuration
# -------------------------------------------------------

MODEL_PATH = os.path.join(
    "saved_model",
    "amr_cnn.keras"
)

CLASS_NAMES = {
    0: "16APSK",
    1: "32APSK",
    2: "64APSK"
}

# -------------------------------------------------------
# Load Model
# -------------------------------------------------------

print("Loading trained model...")

model = load_model(MODEL_PATH)

print("Model loaded successfully.")

# -------------------------------------------------------
# Read Signal
# -------------------------------------------------------

def load_signal(file_path):
    """
    Load complex I/Q signal.
    Supports:
        *.npy
    """

    if not os.path.exists(file_path):
        raise FileNotFoundError(file_path)

    signal = np.load(file_path)

    return signal


# -------------------------------------------------------
# Preprocess Signal
# -------------------------------------------------------

def prepare_signal(signal):
    """
    Convert raw complex signal
    into CNN input.
    """

    signal = preprocess_signal(signal)

    signal = np.expand_dims(signal, axis=0)

    return signal


# -------------------------------------------------------
# Predict Function
# -------------------------------------------------------

def predict_modulation(signal):

    signal = prepare_signal(signal)

    prediction = model.predict(signal, verbose=0)

    class_index = np.argmax(prediction)

    confidence = float(np.max(prediction))

    result = {
        "class_index": int(class_index),
        "modulation": CLASS_NAMES[class_index],
        "confidence": confidence
    }

    return result
# -------------------------------------------------------
# Batch Prediction
# -------------------------------------------------------

def batch_predict(signal_list):
    """
    Predict modulation for multiple signals.
    """

    results = []

    for signal in signal_list:

        result = predict_modulation(signal)

        results.append(result)

    return results


# -------------------------------------------------------
# Pretty Print Result
# -------------------------------------------------------

def print_result(result):

    print("\n==============================")
    print(" Automatic Modulation Recognition")
    print("==============================")
    print(f"Predicted Modulation : {result['modulation']}")
    print(f"Class Index          : {result['class_index']}")
    print(f"Confidence           : {result['confidence']*100:.2f}%")
    print("==============================\n")


# -------------------------------------------------------
# Prediction with Time Measurement
# -------------------------------------------------------

def predict_with_time(signal):

    import time

    start = time.time()

    result = predict_modulation(signal)

    end = time.time()

    result["prediction_time"] = round(end - start, 4)

    return result


# -------------------------------------------------------
# Save Prediction Result
# -------------------------------------------------------

def save_prediction(result,
                    output_file="prediction_result.txt"):

    with open(output_file, "w") as f:

        f.write("Automatic Modulation Recognition\n")
        f.write("--------------------------------\n")
        f.write(f"Predicted Modulation : {result['modulation']}\n")
        f.write(f"Class Index          : {result['class_index']}\n")
        f.write(f"Confidence           : {result['confidence']*100:.2f}%\n")

        if "prediction_time" in result:
            f.write(
                f"Prediction Time      : "
                f"{result['prediction_time']} sec\n"
            )

    print("Prediction saved to", output_file)


# -------------------------------------------------------
# Main Function
# -------------------------------------------------------

if __name__ == "__main__":

    sample_path = "sample.npy"

    try:

        signal = load_signal(sample_path)

        result = predict_with_time(signal)

        print_result(result)

        print(
            f"Prediction Time : "
            f"{result['prediction_time']} seconds"
        )

        save_prediction(result)

    except FileNotFoundError:

        print("Error: sample.npy not found.")

    except Exception as e:

        print("Prediction Failed")

        print(e)