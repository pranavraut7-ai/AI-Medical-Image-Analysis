import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from data_pipeline import load_datasets


MODEL_PATH = "models/chest_xray_mobilenetv2.keras"
OUTPUT_PATH = "outputs/test_predictions.png"


def main():
    _, _, test_dataset, class_names = load_datasets()

    model = tf.keras.models.load_model(MODEL_PATH)

    images, labels = next(iter(test_dataset))

    probabilities = model.predict(images, verbose=0).reshape(-1)
    true_labels = labels.numpy().reshape(-1)

    num_samples = min(9, len(images))

    plt.figure(figsize=(12, 12))

    for index in range(num_samples):
        image = images[index].numpy()
        true_label = int(true_labels[index])
        probability = float(probabilities[index])

        predicted_label = int(probability >= 0.5)

        plt.subplot(3, 3, index + 1)
        plt.imshow(image)

        plt.title(
            f"True: {class_names[true_label]}\n"
            f"Pred: {class_names[predicted_label]} "
            f"({probability:.2f})"
        )

        plt.axis("off")

    plt.suptitle("Chest X-ray Test Predictions", fontsize=16)
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=200)
    plt.close()

    print("=" * 60)
    print("PREDICTION VISUALIZATION COMPLETE")
    print("=" * 60)
    print(f"Images displayed : {num_samples}")
    print(f"Output saved     : {OUTPUT_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()