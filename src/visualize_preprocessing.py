import matplotlib.pyplot as plt

from data_pipeline import load_datasets


def visualize_preprocessed_images():
    """
    Display a small batch of preprocessed X-ray images
    and save the visualization as project evidence.
    """

    train_dataset, _, _, class_names = load_datasets()

    images, labels = next(iter(train_dataset))

    plt.figure(figsize=(12, 8))

    for index in range(6):
        plt.subplot(2, 3, index + 1)

        image = images[index].numpy()
        label = int(labels[index].numpy()[0])

        plt.imshow(image)
        plt.title(
            f"Label: {class_names[label]}\n"
            f"Shape: {image.shape}"
        )
        plt.axis("off")

    plt.tight_layout()

    output_path = "outputs/preprocessing_samples.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")

    print("=" * 60)
    print("PREPROCESSING VISUALIZATION COMPLETE")
    print("=" * 60)
    print(f"Images displayed : 6")
    print(f"Image shape      : {images[0].shape}")
    print(f"Output saved     : {output_path}")
    print("=" * 60)

    plt.show()


if __name__ == "__main__":
    visualize_preprocessed_images()