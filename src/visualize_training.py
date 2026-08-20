import csv
import matplotlib.pyplot as plt


HISTORY_PATH = "outputs/training_history.csv"
OUTPUT_PATH = "outputs/training_history.png"


def main():
    epochs = []
    training_accuracy = []
    validation_accuracy = []

    with open(HISTORY_PATH, "r", newline="") as file:
        reader = csv.DictReader(file)

        for index, row in enumerate(reader, start=1):
            epochs.append(index)
            training_accuracy.append(float(row["accuracy"]))
            validation_accuracy.append(float(row["val_accuracy"]))

    plt.figure(figsize=(10, 5))

    plt.plot(
        epochs,
        training_accuracy,
        marker="o",
        label="Training Accuracy",
    )

    plt.plot(
        epochs,
        validation_accuracy,
        marker="o",
        label="Validation Accuracy",
    )

    plt.title("Model Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.xticks(epochs)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(OUTPUT_PATH, dpi=200)
    plt.close()

    print("=" * 60)
    print("TRAINING VISUALIZATION COMPLETE")
    print("=" * 60)
    print(f"Epochs plotted : {len(epochs)}")
    print(f"Output saved   : {OUTPUT_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()