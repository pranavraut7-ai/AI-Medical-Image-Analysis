import sys
from pathlib import Path

import numpy as np
from PIL import Image
from tensorflow import keras


IMAGE_SIZE = (224, 224)
MODEL_PATH = Path("models/chest_xray_mobilenetv2.keras")


def predict_image(image_path):
    model = keras.models.load_model(MODEL_PATH)

    image = Image.open(image_path).convert("RGB")
    image = image.resize(IMAGE_SIZE)

    image_array = np.array(image, dtype=np.float32) / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    prediction = float(model.predict(image_array, verbose=0)[0][0])

    if prediction >= 0.5:
        label = "PNEUMONIA"
        confidence = prediction
    else:
        label = "NORMAL"
        confidence = 1.0 - prediction

    print("=" * 60)
    print("CHEST X-RAY SINGLE IMAGE PREDICTION")
    print("=" * 60)
    print(f"Image       : {image_path}")
    print(f"Prediction  : {label}")
    print(f"Confidence  : {confidence:.4f}")
    print(f"Probability : {prediction:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python src/predict.py <image_path>")
        sys.exit(1)

    predict_image(Path(sys.argv[1]))