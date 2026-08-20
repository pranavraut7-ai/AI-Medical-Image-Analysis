from pathlib import Path

import tensorflow as tf


# ============================================================
# Configuration
# ============================================================

DATASET_DIR = Path(
    r"D:\Datasets\Medical-Datasets\chest_xray\chest_xray"
)

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42


# ============================================================
# Dataset Loading
# ============================================================

def load_datasets():
    """
    Load the training, validation, and test datasets.

    Images are:
    - resized to 224 x 224
    - loaded as RGB (3 channels)
    - labeled as NORMAL or PNEUMONIA
    """

    train_dir = DATASET_DIR / "train"
    val_dir = DATASET_DIR / "val"
    test_dir = DATASET_DIR / "test"

    train_dataset = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        labels="inferred",
        label_mode="binary",
        color_mode="rgb",
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed=SEED,
    )

    validation_dataset = tf.keras.utils.image_dataset_from_directory(
        val_dir,
        labels="inferred",
        label_mode="binary",
        color_mode="rgb",
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    test_dataset = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        labels="inferred",
        label_mode="binary",
        color_mode="rgb",
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    class_names = train_dataset.class_names

    # Normalize pixel values from [0, 255] to [0, 1].
    normalization_layer = tf.keras.layers.Rescaling(
        1.0 / 255.0
    )

    train_dataset = train_dataset.map(
        lambda images, labels: (
            normalization_layer(images),
            labels,
        ),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    validation_dataset = validation_dataset.map(
        lambda images, labels: (
            normalization_layer(images),
            labels,
        ),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    test_dataset = test_dataset.map(
        lambda images, labels: (
            normalization_layer(images),
            labels,
        ),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    # Prefetch improves pipeline efficiency without loading
    # the entire dataset into memory.
    train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
    validation_dataset = validation_dataset.prefetch(tf.data.AUTOTUNE)
    test_dataset = test_dataset.prefetch(tf.data.AUTOTUNE)

    return (
        train_dataset,
        validation_dataset,
        test_dataset,
        class_names,
    )


# ============================================================
# Pipeline Verification
# ============================================================

def verify_pipeline():
    """
    Load the datasets and verify their basic structure.
    """

    (
        train_dataset,
        validation_dataset,
        test_dataset,
        class_names,
    ) = load_datasets()

    train_images, train_labels = next(iter(train_dataset))
    val_images, val_labels = next(iter(validation_dataset))
    test_images, test_labels = next(iter(test_dataset))

    print("\n" + "=" * 60)
    print("DATA PIPELINE VERIFICATION")
    print("=" * 60)

    print(f"Dataset directory : {DATASET_DIR}")
    print(f"Class names       : {class_names}")

    print("\nTrain batch:")
    print(f"  Images shape     : {train_images.shape}")
    print(f"  Labels shape     : {train_labels.shape}")
    print(
        f"  Pixel range      : "
        f"{tf.reduce_min(train_images).numpy():.4f} "
        f"to "
        f"{tf.reduce_max(train_images).numpy():.4f}"
    )

    print("\nValidation batch:")
    print(f"  Images shape     : {val_images.shape}")
    print(f"  Labels shape     : {val_labels.shape}")

    print("\nTest batch:")
    print(f"  Images shape     : {test_images.shape}")
    print(f"  Labels shape     : {test_labels.shape}")

    print("\nExpected image format:")
    print("  Height           : 224")
    print("  Width            : 224")
    print("  Channels         : 3")
    print("  Pixel range      : 0.0 to 1.0")

    print("\n" + "=" * 60)
    print("PIPELINE VERIFICATION COMPLETE")
    print("=" * 60)


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":
    verify_pipeline()