import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

from data_pipeline import load_datasets


MODEL_PATH = "models/chest_xray_mobilenetv2.keras"


def main():
    train_dataset, validation_dataset, test_dataset, class_names = load_datasets()

    model = tf.keras.models.load_model(MODEL_PATH)

    true_labels = []
    predictions = []

    for images, labels in test_dataset:
        probabilities = model.predict(images, verbose=0).reshape(-1)

        true_labels.extend(labels.numpy().astype(int).reshape(-1))
        predictions.extend((probabilities >= 0.5).astype(int))

    true_labels = np.array(true_labels)
    predictions = np.array(predictions)

    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions, zero_division=0)
    recall = recall_score(true_labels, predictions, zero_division=0)
    f1 = f1_score(true_labels, predictions, zero_division=0)
    matrix = confusion_matrix(true_labels, predictions)

    print()
    print("=" * 60)
    print("TEST SET EVALUATION")
    print("=" * 60)

    print(f"Test images       : {len(true_labels)}")
    print(f"Class names       : {class_names}")
    print()
    print(f"Accuracy          : {accuracy:.4f}")
    print(f"Precision         : {precision:.4f}")
    print(f"Recall            : {recall:.4f}")
    print(f"F1-score          : {f1:.4f}")

    print()
    print("Confusion Matrix")
    print(matrix)

    print()
    print("Classification Report")
    print(
        classification_report(
            true_labels,
            predictions,
            target_names=class_names,
            zero_division=0,
        )
    )

    print("=" * 60)
    print("TEST EVALUATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()