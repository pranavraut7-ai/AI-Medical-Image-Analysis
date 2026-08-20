import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


IMAGE_SIZE = (224, 224)
NUM_CHANNELS = 3


def build_model():
    """
    Build a binary chest X-ray classification model
    using MobileNetV2 transfer learning.
    """

    base_model = tf.keras.applications.MobileNetV2(
        input_shape=IMAGE_SIZE + (NUM_CHANNELS,),
        include_top=False,
        weights="imagenet"
    )

    # Freeze the pretrained feature extractor.
    base_model.trainable = False

    inputs = keras.Input(
        shape=IMAGE_SIZE + (NUM_CHANNELS,),
        name="chest_xray_input"
    )

    # Our data pipeline produces pixels in the range 0 to 1.
    # MobileNetV2 expects inputs scaled approximately from -1 to 1.
    x = layers.Rescaling(
        scale=2.0,
        offset=-1.0,
        name="mobilenetv2_input_scaling"
    )(inputs)

    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D(
        name="global_average_pooling"
    )(x)

    x = layers.Dropout(
        0.30,
        name="dropout"
    )(x)

    outputs = layers.Dense(
        1,
        activation="sigmoid",
        name="pneumonia_probability"
    )(x)

    model = keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="chest_xray_mobilenetv2"
    )

    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=0.0001
        ),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall")
        ]
    )

    return model


if __name__ == "__main__":
    model = build_model()

    print("=" * 70)
    print("MODEL BUILD VERIFICATION")
    print("=" * 70)

    print(f"Model name   : {model.name}")
    print(f"Input shape  : {model.input_shape}")
    print(f"Output shape : {model.output_shape}")
    print()

    model.summary()

    print()
    print("=" * 70)
    print("MODEL BUILD COMPLETE")
    print("=" * 70)
    print("Architecture : MobileNetV2 Transfer Learning")
    print("Input        : 224 x 224 x 3")
    print("Output       : 1 sigmoid probability")
    print("Classes      : NORMAL / PNEUMONIA")
    print("Base model   : Frozen")
    print("Input scaling: 0-1 to -1-1")
    print("=" * 70)