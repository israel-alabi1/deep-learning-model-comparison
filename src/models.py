"""Neural-network model definitions."""

from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    Input, Dense, SimpleRNN, LSTM, GRU,
    Conv2D, MaxPooling2D, Flatten,
)
from tensorflow.keras.optimizers import Adam


def build_sequence_model(model_type, input_shape, learning_rate=1e-3):
    model = Sequential([Input(shape=input_shape)])
    if model_type == "ffnn":
        model.add(Flatten())
        model.add(Dense(128, activation="relu"))
        model.add(Dense(64, activation="relu"))
    elif model_type == "rnn":
        model.add(SimpleRNN(32, activation="tanh"))
    elif model_type == "lstm":
        model.add(LSTM(32, activation="tanh"))
    elif model_type == "gru":
        model.add(GRU(32, activation="tanh"))
    else:
        raise ValueError("Unknown model_type")
    model.add(Dense(1))
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=["mae"],
    )
    return model


def build_baseline_cnn(input_shape, num_classes, learning_rate=1e-3):
    model = Sequential([
        Input(shape=input_shape),
        Conv2D(32, (3, 3), activation="relu"),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation="relu"),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation="relu"),
        Dense(num_classes, activation="softmax"),
    ])
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
