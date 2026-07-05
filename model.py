"""
model.py
CNN Model for Automatic Modulation Recognition (AMR)
Supports DVB-S2X APSK Modulations:
    - 16APSK
    - 32APSK
    - 64APSK
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv1D,
    BatchNormalization,
    MaxPooling1D,
    Dropout,
    Flatten,
    Dense
)
from tensorflow.keras.optimizers import Adam


INPUT_SHAPE = (128, 2)
NUM_CLASSES = 3


def build_model():

    model = Sequential(name="AMR_CNN")

    # Block 1
    model.add(
        Conv1D(
            filters=64,
            kernel_size=3,
            activation="relu",
            padding="same",
            input_shape=INPUT_SHAPE
        )
    )
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))

    # Block 2
    model.add(
        Conv1D(
            filters=128,
            kernel_size=3,
            activation="relu",
            padding="same"
        )
    )
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))

    # Block 3
    model.add(
        Conv1D(
            filters=256,
            kernel_size=3,
            activation="relu",
            padding="same"
        )
    )
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))

    # Block 4
    model.add(
        Conv1D(
            filters=512,
            kernel_size=3,
            activation="relu",
            padding="same"
        )
    )
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))

    # Classification Head
    model.add(Flatten())

    model.add(Dense(512, activation="relu"))
    model.add(Dropout(0.5))

    model.add(Dense(256, activation="relu"))
    model.add(Dropout(0.3))

    model.add(Dense(128, activation="relu"))

    model.add(
        Dense(
            NUM_CLASSES,
            activation="softmax",
            name="Prediction"
        )
    )

    optimizer = Adam(
        learning_rate=0.001
    )

    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=[
            "accuracy"
        ]
    )

    return model


def print_model_summary():

    model = build_model()

    model.summary()


if __name__ == "__main__":

    model = build_model()

    model.summary()

    tf.keras.utils.plot_model(
        model,
        show_shapes=True,
        dpi=120,
        to_file="cnn_architecture.png"
    )