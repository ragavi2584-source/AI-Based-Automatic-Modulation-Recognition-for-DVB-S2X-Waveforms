"""
train.py
Train CNN for Automatic Modulation Recognition
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau
)
from tensorflow.keras.utils import to_categorical

from model import build_model

# -------------------------------------------------
# Configuration
# -------------------------------------------------

DATASET_DIR = "dataset"

X_PATH = os.path.join(DATASET_DIR, "train_x_processed.npy")
Y_PATH = os.path.join(DATASET_DIR, "train_y_processed.npy")

MODEL_DIR = "saved_model"

MODEL_NAME = "amr_cnn.keras"

BATCH_SIZE = 64
EPOCHS = 50

# -------------------------------------------------
# Create Folder
# -------------------------------------------------

os.makedirs(MODEL_DIR, exist_ok=True)

# -------------------------------------------------
# Load Dataset
# -------------------------------------------------

print("Loading dataset...")

X = np.load(X_PATH)
y = np.load(Y_PATH)

print("Dataset Shape :", X.shape)
print("Labels Shape  :", y.shape)

# -------------------------------------------------
# One-Hot Encoding
# -------------------------------------------------

y = to_categorical(y, num_classes=3)

# -------------------------------------------------
# Train/Test Split
# -------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    shuffle=True
)

print("Training Samples :", len(X_train))
print("Testing Samples  :", len(X_test))

# -------------------------------------------------
# Build Model
# -------------------------------------------------

model = build_model()

# -------------------------------------------------
# Callbacks
# -------------------------------------------------

checkpoint = ModelCheckpoint(
    filepath=os.path.join(MODEL_DIR, MODEL_NAME),
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1
)

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=8,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=4,
    verbose=1
)

# -------------------------------------------------
# Train
# -------------------------------------------------

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[
        checkpoint,
        early_stop,
        reduce_lr
    ],
    verbose=1
)

# -------------------------------------------------
# Evaluation
# -------------------------------------------------

loss, accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=0
)

print("\nFinal Test Accuracy : {:.2f}%".format(accuracy * 100))
print("Final Test Loss     : {:.4f}".format(loss))

# -------------------------------------------------
# Save Final Model
# -------------------------------------------------

model.save(
    os.path.join(
        MODEL_DIR,
        "final_model.keras"
    )
)

print("Model Saved Successfully.")

# -------------------------------------------------
# Plot Accuracy
# -------------------------------------------------

plt.figure(figsize=(8,5))

plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")

plt.title("Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.grid(True)

plt.savefig("accuracy.png")

plt.close()

# -------------------------------------------------
# Plot Loss
# -------------------------------------------------

plt.figure(figsize=(8,5))

plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")

plt.title("Model Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.grid(True)

plt.savefig("loss.png")

plt.close()

print("Training Completed Successfully.")