from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image
from tensorflow import keras


IMAGE_SIZE = (224, 224)
MODEL_PATH = Path("models/chest_xray_mobilenetv2.keras")


st.set_page_config(
    page_title="AI Medical Image Analysis",
    page_icon="🩻",
    layout="wide",
    initial_sidebar_state="collapsed",
)


@st.cache_resource
def load_model():
    return keras.models.load_model(MODEL_PATH)


def predict_image(image):
    image = image.convert("RGB")
    image = image.resize(IMAGE_SIZE)

    image_array = np.array(image, dtype=np.float32) / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    model = load_model()

    prediction = float(model.predict(image_array, verbose=0)[0][0])

    if prediction >= 0.5:
        label = "PNEUMONIA"
        confidence = prediction
    else:
        label = "NORMAL"
        confidence = 1.0 - prediction

    return label, confidence, prediction


st.markdown(
    """
    <style>
        .stApp {
            background: #0b1016;
        }

        [data-testid="stHeader"] {
            background: rgba(11, 16, 22, 0.85);
        }

        .block-container {
            max-width: 1400px;
            padding-top: 2.5rem;
            padding-bottom: 3rem;
        }

        .hero {
            padding: 1.5rem 0 2.2rem 0;
        }

        .eyebrow {
            color: #72d6e8;
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.16em;
            text-transform: uppercase;
            margin-bottom: 0.75rem;
        }

        .hero h1 {
            color: #f4f7fa;
            font-size: 3rem;
            line-height: 1.05;
            margin: 0;
            font-weight: 700;
        }

        .hero p {
            color: #aeb9c5;
            font-size: 1.05rem;
            margin-top: 0.8rem;
            max-width: 720px;
        }

        .panel {
            background: #121923;
            border: 1px solid #25313d;
            border-radius: 18px;
            padding: 1.35rem;
        }

        .upload-panel,
        .analysis-panel,
        .result-panel,
        .detail-panel {
            min-height: 330px;
        }

        .panel-title {
            color: #f4f7fa;
            font-size: 1.05rem;
            font-weight: 650;
            margin-bottom: 0.35rem;
        }

        .panel-subtitle {
            color: #8492a0;
            font-size: 0.88rem;
            margin-bottom: 1.2rem;
        }

        .xray-placeholder {
            min-height: 220px;
            border: 1px dashed #3a4a59;
            border-radius: 14px;
            display: flex;
            align-items: center;
            justify-content: center;
            text-align: center;
            color: #71808e;
            background: #0d141c;
            padding: 2rem;
        }

        .xray-placeholder strong {
            color: #aeb9c5;
            font-weight: 600;
        }

        .analysis-placeholder {
            min-height: 220px;
            display: flex;
            flex-direction: column;
            justify-content: center;
        }

        .status-label {
            color: #72d6e8;
            font-size: 0.76rem;
            font-weight: 700;
            letter-spacing: 0.14em;
            text-transform: uppercase;
        }

        .analysis-value {
            color: #f4f7fa;
            font-size: 2.15rem;
            font-weight: 750;
            line-height: 1.1;
            margin: 0.55rem 0 0.85rem 0;
        }

        .analysis-note {
            color: #9eabb7;
            font-size: 0.88rem;
            line-height: 1.6;
        }

        .result-status {
            color: #72d6e8;
            font-size: 0.76rem;
            font-weight: 700;
            letter-spacing: 0.14em;
            text-transform: uppercase;
            margin-bottom: 1.2rem;
        }

        .prediction-label,
        .detail-label {
            color: #82909e;
            font-size: 0.76rem;
            font-weight: 650;
            letter-spacing: 0.12em;
            text-transform: uppercase;
        }

        .prediction-value {
            color: #f4f7fa;
            font-size: 2.45rem;
            line-height: 1.05;
            font-weight: 750;
            margin: 0.45rem 0 0.75rem 0;
        }

        .prediction-confidence {
            color: #72d6e8;
            font-size: 1rem;
            font-weight: 650;
        }

        .result-description {
            color: #9eabb7;
            font-size: 0.88rem;
            line-height: 1.6;
            margin-top: 1.1rem;
            max-width: 600px;
        }

        .detail-row {
            padding: 1rem 0;
            border-bottom: 1px solid #25313d;
        }

        .detail-value {
            color: #f4f7fa;
            font-size: 1.35rem;
            font-weight: 650;
            margin-top: 0.3rem;
        }

        .threshold-note {
            color: #9eabb7;
            font-size: 0.88rem;
            line-height: 1.6;
            margin-top: 1rem;
        }

        .metric-card {
            background: #101720;
            border: 1px solid #25313d;
            border-radius: 14px;
            padding: 1rem;
        }

        .metric-name {
            color: #7f8c99;
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
        }

        .metric-value {
            color: #eef3f7;
            font-size: 1.1rem;
            font-weight: 650;
            margin-top: 0.35rem;
        }

        .disclaimer {
            margin-top: 1.5rem;
            padding: 0.9rem 1rem;
            border-left: 3px solid #536575;
            background: #0f151d;
            color: #8997a5;
            font-size: 0.78rem;
            line-height: 1.5;
            border-radius: 0 10px 10px 0;
        }

        div.stButton > button {
            background: #121923;
            color: #f4f7fa;
            border: 1px solid #3a4a59;
            border-radius: 10px;
            min-height: 42px;
            font-weight: 650;
        }

        div.stButton > button:hover {
            border-color: #72d6e8;
            color: #72d6e8;
        }

        [data-testid="stFileUploader"] {
            background: #151922;
            border-radius: 10px;
        }

        footer {
            visibility: hidden;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


st.markdown(
    """
    <section class="hero">
        <div class="eyebrow">AI • Medical Imaging</div>
        <h1>AI Medical Image Analysis</h1>
        <p>
            Chest X-ray pneumonia classification powered by
            MobileNetV2 transfer learning.
        </p>
    </section>
    """,
    unsafe_allow_html=True,
)


upload_col, analysis_col = st.columns([1.15, 0.85], gap="large")


with upload_col:

    st.html(
        """
        <div class="panel upload-panel">
            <div class="panel-title">Chest X-ray</div>
            <div class="panel-subtitle">
                Upload an image to prepare it for AI analysis.
            </div>
        </div>
        """
    )

    uploaded_file = st.file_uploader(
        "Upload chest X-ray",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
    )

    if uploaded_file is None:
        st.html(
            """
            <div class="xray-placeholder">
                <div>
                    <strong>Upload a chest X-ray</strong><br>
                    JPG, JPEG or PNG
                </div>
            </div>
            """
        )
    else:
        image = Image.open(uploaded_file).convert("RGB")

        st.image(
            image,
            caption="Uploaded chest X-ray",
            width="stretch",
        )


with analysis_col:

    if "prediction" in st.session_state:
        analysis_status = "Analysis Complete"
        analysis_value = "Result available"
        analysis_note = (
            "The trained model has generated a classification "
            "from the uploaded chest X-ray."
        )

    elif uploaded_file is not None:
        analysis_status = "Analysis Ready"
        analysis_value = "X-ray loaded"
        analysis_note = (
            "Run the AI analysis to generate the model prediction."
        )

    else:
        analysis_status = "Awaiting Input"
        analysis_value = "Awaiting analysis"
        analysis_note = (
            "Upload an X-ray to begin the analysis workflow."
        )

    st.html(
        f"""
        <div class="panel analysis-panel">
            <div class="panel-title">AI Analysis</div>
            <div class="panel-subtitle">
                Model result will appear here after analysis.
            </div>
            <div class="analysis-placeholder">
                <div class="status-label">{analysis_status}</div>
                <div class="analysis-value">{analysis_value}</div>
                <div class="analysis-note">{analysis_note}</div>
            </div>
        </div>
        """
    )


st.markdown("<br>", unsafe_allow_html=True)


if uploaded_file is not None:

    analyze_col, spacer_col = st.columns([0.3, 0.7])

    with analyze_col:

        analyze_button = st.button(
            "Run AI Analysis",
            use_container_width=True,
        )

    if analyze_button:

        with st.spinner("Analyzing chest X-ray..."):

            try:

                image = Image.open(uploaded_file).convert("RGB")

                label, confidence, probability = predict_image(image)

                st.session_state["prediction"] = label
                st.session_state["confidence"] = confidence
                st.session_state["probability"] = probability

                st.rerun()

            except Exception as error:

                st.error(
                    f"Unable to analyze the image. Error: {error}"
                )


if "prediction" in st.session_state:

    prediction = st.session_state["prediction"]
    confidence = st.session_state["confidence"]
    probability = st.session_state["probability"]

    result_col, details_col = st.columns(
        [1.15, 0.85],
        gap="large",
    )

    with result_col:

        st.html(
            f"""
            <div class="panel result-panel">
                <div class="result-status">
                    Analysis Complete
                </div>

                <div class="prediction-label">
                    Model Prediction
                </div>

                <div class="prediction-value">
                    {prediction}
                </div>

                <div class="prediction-confidence">
                    Confidence: {confidence:.2%}
                </div>

                <div class="result-description">
                    Classification generated from the uploaded chest
                    X-ray using the trained MobileNetV2 model.
                </div>
            </div>
            """
        )

    with details_col:

        st.html(
            f"""
            <div class="panel detail-panel">
                <div class="panel-title">
                    Prediction Details
                </div>

                <div class="panel-subtitle">
                    Model output from the current analysis.
                </div>

                <div class="detail-row">
                    <div class="detail-label">
                        Pneumonia Probability
                    </div>

                    <div class="detail-value">
                        {probability:.2%}
                    </div>
                </div>

                <div class="detail-row">
                    <div class="detail-label">
                        Classification Threshold
                    </div>

                    <div class="detail-value">
                        0.50
                    </div>
                </div>

                <div class="threshold-note">
                    A probability of 0.50 or higher is classified as
                    PNEUMONIA. Values below 0.50 are classified as NORMAL.
                </div>
            </div>
            """
        )


st.markdown("<br>", unsafe_allow_html=True)


st.html(
    """
    <div class="panel">
        <div class="panel-title">
            Model Information
        </div>

        <div class="panel-subtitle">
            Technical context for the current classification system.
        </div>
    </div>
    """
)


metric_cols = st.columns(4)

metrics = [
    ("Architecture", "MobileNetV2"),
    ("Learning", "Transfer Learning"),
    ("Input", "224 × 224 × 3"),
    ("Task", "Binary Classification"),
]

for column, (name, value) in zip(metric_cols, metrics):

    with column:

        st.html(
            f"""
            <div class="metric-card">
                <div class="metric-name">
                    {name}
                </div>

                <div class="metric-value">
                    {value}
                </div>
            </div>
            """
        )


st.html(
    """
    <div class="disclaimer">
        Educational and research prototype only. This system is not a
        clinically validated medical device and must not be used for
        diagnosis or treatment decisions.
    </div>
    """
)