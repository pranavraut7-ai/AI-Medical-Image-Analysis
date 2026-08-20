import csv
from pathlib import Path

import tensorflow as tf

from data_pipeline import load_datasets
from model import build_model


MODEL_PATH = Path("models/chest_xray_mobilenetv2.keras")
HISTORY_PATH = Path("outputs/training_history.csv")

EPOCHS = 5


def calculate_class_weights():
    """
    Calculate balanced class weights from the training dataset.

    Class 0 = NORMAL
    Class 1 = PNEUMONIA
    """

    normal_count = 1341
    pneumonia_count = 3875

    total_samples = normal_count + pneumonia_count
    number_of_classes = 2

    normal_weight = total_samples / (
        number_of_classes * normal_count
    )

    pneumonia_weight = total_samples / (
        number_of_classes * pneumonia_count
    )

    return {
        0: normal_weight,
        1: pneumonia_weight,
    }


def save_training_history(history):
    """
    Save training metrics to a CSV file.
    """

    HISTORY_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    history_data = history.history

    metric_names = list(history_data.keys())

    with open(
        HISTORY_PATH,
        "w",
        newline="",
        encoding="utf-8"
    ) as csv_file:

        writer = csv.writer(csv_file)

        writer.writerow(
            ["epoch"] + metric_names
        )

        number_of_epochs = len(
            history_data[metric_names[0]]
        )

        for epoch in range(number_of_epochs):
            row = [epoch + 1]

            for metric_name in metric_names:
                row.append(
                    history_data[metric_name][epoch]
                )

            writer.writerow(row)


def train_model():
    """
    Train the MobileNetV2 chest X-ray classifier.
    """

    print("=" * 70)
    print("LOADING DATA")
    print("=" * 70)

    train_dataset, validation_dataset, _, class_names = (
        load_datasets()
    )

    print()
    print(f"Class names : {class_names}")

    class_weights = calculate_class_weights()

    print()
    print("=" * 70)
    print("CLASS WEIGHTS")
    print("=" * 70)

    print(
        f"NORMAL weight     : "
        f"{class_weights[0]:.4f}"
    )

    print(
        f"PNEUMONIA weight  : "
        f"{class_weights[1]:.4f}"
    )

    print()
    print("=" * 70)
    print("BUILDING MODEL")
    print("=" * 70)

    model = build_model()

    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=2,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(MODEL_PATH),
            monitor="val_loss",
            save_best_only=True,
            verbose=1
        )
    ]

    print()
    print("=" * 70)
    print("STARTING TRAINING")
    print("=" * 70)

    print(f"Epochs       : {EPOCHS}")
    print(f"Model output : {MODEL_PATH}")
    print()

    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=EPOCHS,
        class_weight=class_weights,
        callbacks=callbacks
    )

    save_training_history(history)

    print()
    print("=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)

    print(
        f"Model saved   : {MODEL_PATH}"
    )

    print(
        f"History saved : {HISTORY_PATH}"
    )

    print(
        f"Epochs run    : "
        f"{len(history.history['loss'])}"
    )

    print("=" * 70)


if __name__ == "__main__":
    train_model()