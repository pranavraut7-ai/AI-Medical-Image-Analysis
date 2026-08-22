# AI Based Medical Image Analysis – Chest X-Ray Pneumonia Detection

An AI-based medical image classification system that uses **MobileNetV2 transfer learning** to classify chest X-ray images as **NORMAL** or **PNEUMONIA**.

The project demonstrates an end-to-end machine-learning workflow covering data preprocessing, transfer learning, model training, evaluation, single-image inference, interactive UI development, GitHub version control, and public deployment.

> **Medical Disclaimer:** This project is developed for educational, engineering, research, and portfolio purposes only. It is not a clinically validated medical device and must not be used for diagnosis, treatment, or medical decision-making.

---

## Project Overview

The system takes a chest X-ray image as input and predicts whether the image belongs to one of two classes:

- **NORMAL**
- **PNEUMONIA**

The project follows this workflow:

    Chest X-ray Dataset
            ↓
    Data Loading & Preprocessing
            ↓
    MobileNetV2 Transfer Learning
            ↓
    Model Training
            ↓
    Model Evaluation
            ↓
    Single Image Inference
            ↓
    Streamlit Application
            ↓
    Public Deployment

The project is intended to demonstrate practical AI engineering rather than clinical deployment.

---

## Problem Statement

Chest X-ray images contain visual patterns that can be analyzed using computer vision and deep-learning techniques.

The objective of this project is to build a binary image-classification system that learns from labeled chest X-ray images and predicts whether an input image is classified as **NORMAL** or **PNEUMONIA**.

The system uses a public chest X-ray dataset and a pretrained deep-learning architecture to demonstrate the complete development workflow.

---

## Objectives

The main objectives of the project are:

- Build a reproducible medical image classification pipeline.
- Load and preprocess chest X-ray images.
- Normalize image data for deep-learning inference.
- Apply MobileNetV2 transfer learning.
- Handle class imbalance using class weights.
- Train and validate the classification model.
- Evaluate the model on a separate test set.
- Generate useful evaluation and visualization outputs.
- Implement single-image inference.
- Build an interactive Streamlit application.
- Deploy the application publicly.
- Document the complete engineering workflow.

---

## Key Features

- Chest X-ray binary classification
- MobileNetV2 transfer learning
- Image preprocessing and normalization
- Class imbalance handling
- Early stopping
- Best-model checkpointing
- Accuracy, precision, recall and F1-score evaluation
- Confusion matrix generation
- Classification report generation
- Single-image inference
- Prediction probability and confidence
- Interactive Streamlit interface
- Public Streamlit deployment

---

## Technology Stack

| Category | Technology |
|---|---|
| Programming Language | Python |
| Deep Learning | TensorFlow, Keras |
| Model Architecture | MobileNetV2 |
| Computer Vision | OpenCV, Pillow |
| Numerical Computing | NumPy |
| Evaluation | Scikit-learn |
| Visualization | Matplotlib |
| Application | Streamlit |
| Development | Visual Studio Code |
| Version Control | Git, GitHub |
| Deployment | Streamlit Community Cloud |

---

## Dataset

The project uses a public chest X-ray dataset containing two classes:

- **NORMAL**
- **PNEUMONIA**

Verified class distribution:

| Class | Images |
|---|---:|
| NORMAL | 1,341 |
| PNEUMONIA | 3,875 |
| **Total** | **5,856** |

Verified dataset split:

| Split | Images |
|---|---:|
| Training | 5,216 |
| Validation | 16 |
| Test | 624 |
| **Total** | **5,856** |

The project uses the existing train, validation, and test directories rather than creating a new split in the implementation.

The dataset itself is not included in the GitHub repository.

---

## Dataset Structure

The expected dataset structure is:

    chest_xray/
    │
    ├── train/
    │   ├── NORMAL/
    │   └── PNEUMONIA/
    │
    ├── val/
    │   ├── NORMAL/
    │   └── PNEUMONIA/
    │
    └── test/
        ├── NORMAL/
        └── PNEUMONIA/

The local dataset path is configured through `DATASET_DIR` in:

    src/data_pipeline.py

---

## Data Pipeline

The data pipeline is implemented in:

    src/data_pipeline.py

TensorFlow's `image_dataset_from_directory()` is used to load the images.

### Configuration

| Parameter | Value |
|---|---|
| Label Mode | Binary |
| Color Mode | RGB |
| Image Size | 224 × 224 |
| Channels | 3 |
| Batch Size | 32 |
| Random Seed | 42 |
| Training Shuffle | Enabled |
| Validation Shuffle | Disabled |
| Test Shuffle | Disabled |

---

## Image Preprocessing

Each image follows the following preprocessing workflow:

    Original Chest X-ray
            ↓
    Convert to RGB
            ↓
    Resize to 224 × 224
            ↓
    Convert to numerical array
            ↓
    Normalize 0–255 → 0–1
            ↓
    Model Input

The TensorFlow data pipeline also uses `tf.data.AUTOTUNE` for prefetching.

No image augmentation is implemented in the current version.

---

## Model Architecture

The project uses **MobileNetV2 transfer learning**.

MobileNetV2 is initialized using ImageNet pretrained weights and used as a frozen feature extractor.

    MobileNetV2
        │
        ├── ImageNet pretrained weights
        ├── include_top = False
        └── Frozen feature extractor

The pretrained feature extractor remains frozen during the current training process.

### Classification Head

    MobileNetV2 Feature Extractor
                ↓
    Global Average Pooling
                ↓
    Dropout (0.30)
                ↓
    Dense Layer (1 Unit)
                ↓
    Sigmoid Activation
                ↓
    PNEUMONIA Probability

### Model Input

    224 × 224 × 3

### Model Output

    1 sigmoid probability

The sigmoid output represents the predicted probability of the **PNEUMONIA** class.

---

## MobileNetV2 Input Scaling

The data pipeline initially normalizes image values:

    0–255 → 0–1

The model then applies the additional MobileNetV2 scaling:

    0–1 → -1–1

The transformation is kept inside the model so that preprocessing remains consistent during training and inference.

---

## Training

Training is implemented in:

    src/train.py

### Training Configuration

| Parameter | Value |
|---|---|
| Architecture | MobileNetV2 Transfer Learning |
| Input Size | 224 × 224 × 3 |
| Optimizer | Adam |
| Learning Rate | 0.0001 |
| Loss Function | Binary Crossentropy |
| Epochs | 5 |
| Batch Size | 32 |
| Dropout | 0.30 |
| Base Model | Frozen |

---

## Class Imbalance Handling

The training dataset contains more PNEUMONIA images than NORMAL images.

Training class counts:

    NORMAL     : 1,341
    PNEUMONIA  : 3,875

Class weights are calculated from the training distribution and passed to Keras during training.

This gives the minority NORMAL class greater influence during model optimization.

---

## Training Callbacks

### Early Stopping

The model monitors:

    val_loss

Configuration:

    patience = 2
    restore_best_weights = True

This allows training to stop when validation loss stops improving while restoring the best observed model weights.

### Model Checkpoint

The best model according to validation loss is saved as:

    models/chest_xray_mobilenetv2.keras

The checkpoint uses:

    save_best_only = True

---

## Training History

Training history is saved to:

    outputs/training_history.csv

Training visualization is generated during the visualization stage.

The training history can be used to inspect:

- Training accuracy
- Validation accuracy
- Training behavior across epochs

---

## Model Evaluation

Model evaluation is implemented in:

    src/evaluate.py

The saved model is evaluated against the test dataset containing **624 images**.

The evaluation calculates:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
- Classification Report

### Classification Threshold

The current classification threshold is:

    0.50

Decision logic:

    Probability >= 0.50
            ↓
       PNEUMONIA

    Probability < 0.50
            ↓
         NORMAL

---

## Verified Test Results

The verified test-set results are:

| Metric | Result |
|---|---:|
| Test Images | **624** |
| Accuracy | **87.57%** |
| Precision | **86.79%** |
| Recall | **94.36%** |
| F1-score | **90.42%** |

The model correctly classified:

- 178 NORMAL images
- 368 PNEUMONIA images

These results represent performance on the project's test dataset only and should not be interpreted as clinical performance.

---

## Single Image Inference

Single-image inference is implemented in:

    src/predict.py

The inference workflow is:

    Input X-ray
          ↓
    Convert to RGB
          ↓
    Resize to 224 × 224
          ↓
    Normalize 0–255 → 0–1
          ↓
    Load trained model
          ↓
    Generate sigmoid probability
          ↓
    Apply 0.50 threshold
          ↓
    NORMAL / PNEUMONIA
          ↓
    Confidence + Probability

The prediction script reports:

- Input image path
- Predicted class
- Confidence
- PNEUMONIA probability

### Verified Example

    Prediction  : NORMAL
    Confidence  : 0.6413
    Probability : 0.3587

The probability represents the predicted probability of the PNEUMONIA class.

---

## Streamlit Application

The project includes an interactive Streamlit application:

    app.py

The UI was developed after the core machine-learning pipeline was completed.

The application allows the user to:

1. Upload a chest X-ray.
2. Preview the image.
3. Run AI analysis.
4. View the predicted class.
5. View confidence.
6. View pneumonia probability.
7. View model information.

This extends the project from a trained machine-learning model into a usable AI application.

---

## Application Workflow

    User
      ↓
    Upload Chest X-ray
      ↓
    Image Preview
      ↓
    Run AI Analysis
      ↓
    Load Trained Model
      ↓
    Preprocess Image
      ↓
    Generate Probability
      ↓
    Apply Classification Threshold
      ↓
    Display Result

---

## Live Demo

The application is publicly deployed using **Streamlit Community Cloud**.

### Live Application

https://ai-medical-image-analysis-kxbuyy5vq8oxzks3jnr8gb.streamlit.app/

The deployed application was successfully tested using a chest X-ray image.

Deployment environment:

    Python 3.12

---

## Project Structure

    Project-4-AI-Medical-Image-Analysis/
    │
    ├── src/
    │   ├── data_pipeline.py
    │   ├── model.py
    │   ├── train.py
    │   ├── evaluate.py
    │   ├── predict.py
    │   ├── visualize_preprocessing.py
    │   ├── visualize_training.py
    │   └── visualize_predictions.py
    │
    ├── models/
    │   └── chest_xray_mobilenetv2.keras
    │
    ├── outputs/
    │
    ├── app.py
    ├── requirements.txt
    ├── .gitignore
    └── README.md

The local dataset is excluded from the repository.

---

## Important Files

| File | Purpose |
|---|---|
| `app.py` | Streamlit application |
| `src/data_pipeline.py` | Dataset loading and preprocessing |
| `src/model.py` | MobileNetV2 model architecture |
| `src/train.py` | Model training |
| `src/evaluate.py` | Model evaluation |
| `src/predict.py` | Single-image inference |
| `requirements.txt` | Project dependencies |
| `.gitignore` | Git exclusions |
| `README.md` | Project documentation |

---

## Requirements

The project dependencies are defined in:

    requirements.txt

Core libraries include:

    tensorflow
    opencv-python
    numpy
    matplotlib
    scikit-learn
    pillow
    streamlit

---

## Installation

### Windows

Create a virtual environment:

    python -m venv .venv

Activate the environment:

    .venv\Scripts\Activate.ps1

Install dependencies:

    pip install -r requirements.txt

### macOS / Linux

Create a virtual environment:

    python3 -m venv .venv

Activate the environment:

    source .venv/bin/activate

Install dependencies:

    pip install -r requirements.txt

---

## Dataset Configuration

The dataset is not included in the GitHub repository.

After downloading and extracting the dataset, configure the local path in:

    src/data_pipeline.py

Example:

    DATASET_DIR = Path(
        r"YOUR_LOCAL_CHEST_XRAY_DATASET_PATH"
    )

Replace the placeholder with the actual local `chest_xray` directory.

---

## How to Run

### 1. Verify the Data Pipeline

    python src/data_pipeline.py

This verifies dataset loading, class names, image shapes, label shapes, and pixel range.

### 2. Verify the Model

    python src/model.py

This verifies the model architecture, input shape, output shape, MobileNetV2 configuration, and frozen base model.

### 3. Train the Model

    python src/train.py

This performs dataset loading, class-weight calculation, model training, early stopping, checkpointing, and training-history saving.

### 4. Evaluate the Model

    python src/evaluate.py

This produces the evaluation metrics, confusion matrix, and classification report.

### 5. Run Single Image Prediction

    python src/predict.py "<PATH_TO_XRAY_IMAGE>"

Example:

    python src/predict.py "D:\Datasets\Medical-Datasets\chest_xray\chest_xray\test\NORMAL\IM-0001-0001.jpeg"

### 6. Run the Streamlit Application

    streamlit run app.py

---

## Deployment

The application is deployed using **Streamlit Community Cloud**.

Deployment configuration:

    Repository : pranavraut7-ai/AI-Medical-Image-Analysis
    Branch     : main
    Main File  : app.py
    Python     : 3.12

### Deployment Compatibility Issue

During the initial deployment attempt, the environment used Python 3.14 and TensorFlow dependency installation failed.

The deployment was recreated using Python 3.12.

The application then successfully installed its dependencies, launched, and completed cloud inference testing.

This was a practical dependency and Python-version compatibility issue rather than a model-architecture problem.

---

## Cloud Inference Test

After deployment, the public application was tested using a chest X-ray image.

Verified workflow:

    Image Upload        → PASS
    Image Preview       → PASS
    Model Loading       → PASS
    Inference           → PASS
    Prediction Display  → PASS
    Confidence Display  → PASS
    Probability Display → PASS

The complete cloud inference workflow therefore works successfully.

---

## Project Development Workflow

The project was developed progressively:

    Project Setup
          ↓
    Dataset & Data Pipeline
          ↓
    Model Development
          ↓
    Training
          ↓
    Evaluation
          ↓
    Prediction
          ↓
    Visualization
          ↓
    Streamlit UI
          ↓
    GitHub Repository
          ↓
    Cloud Deployment
          ↓
    Cloud Testing
          ↓
    Documentation

The UI was intentionally developed after the core AI system was functional.

This demonstrates progression from a machine-learning implementation toward a complete AI application.

---

## Proof of Work

The project produces practical evidence across the development lifecycle.

### Data

- Dataset verification
- Class distribution
- Image loading
- Image shape verification

### Model

- Model summary
- MobileNetV2 configuration
- Frozen base model
- Classification head

### Training

- Training logs
- Epoch results
- Validation behavior
- Best-model checkpoint

### Evaluation

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix
- Classification report

### Inference

- Single-image prediction
- Probability
- Confidence
- Test prediction visualization

### Application

- Streamlit interface
- Uploaded X-ray
- AI prediction
- Public cloud deployment
- Successful cloud inference

---

## Industry-Oriented Workflow

The project follows a practical AI engineering lifecycle:

    Problem Definition
          ↓
    Dataset Selection
          ↓
    Data Pipeline
          ↓
    Preprocessing
          ↓
    Model Development
          ↓
    Training
          ↓
    Evaluation
          ↓
    Inference
          ↓
    Application Development
          ↓
    Cloud Deployment
          ↓
    Testing
          ↓
    Documentation

The objective is to demonstrate the complete engineering process required to move from an AI problem to a working application.

---

## Limitations

This project is an educational and engineering proof-of-work rather than a clinical diagnostic system.

### Dataset Dependency

The model was trained on a specific public chest X-ray dataset.

Performance may differ when applied to:

- Different datasets
- Different hospitals
- Different imaging devices
- Different patient populations
- Different acquisition conditions

### Small Validation Set

The provided validation set is relatively small compared with the training and test datasets.

### Frozen Feature Extractor

The current implementation keeps the MobileNetV2 feature extractor frozen.

Fine-tuning is not implemented.

### Binary Classification

The model only predicts:

    NORMAL
    PNEUMONIA

It does not classify other diseases or specific pneumonia subtypes.

### No Clinical Validation

The model has not been clinically validated.

The reported metrics are dataset-level machine-learning results.

### Generalization

Performance on this dataset does not guarantee equivalent performance on external clinical datasets.

---

## Future Improvements

Potential future improvements include:

- Fine-tuning selected MobileNetV2 layers.
- Testing additional transfer-learning architectures.
- Controlled image augmentation.
- Evaluation on an independent external dataset.
- Grad-CAM-based explainability.
- Improved image-quality validation.
- Experiment tracking.
- Model versioning.
- Automated testing.
- Improved deployment monitoring.

These are future possibilities and are not part of the current implementation.

---

## Learning Outcomes

This project provided practical experience in:

- Computer Vision
- Deep Learning
- TensorFlow
- Keras
- Transfer Learning
- MobileNetV2
- Medical Image Classification
- Image Preprocessing
- Binary Classification
- Class Imbalance Handling
- Model Training
- Early Stopping
- Model Checkpointing
- Model Evaluation
- Confusion Matrix Analysis
- Single Image Inference
- Streamlit
- Git
- GitHub
- Cloud Deployment
- AI Application Development

---

## Final Results

    Dataset Images : 5,856
    Test Images    : 624

    Architecture   : MobileNetV2
    Learning       : Transfer Learning
    Input          : 224 × 224 × 3
    Task           : Binary Classification

    Accuracy       : 87.57%
    Precision      : 86.79%
    Recall         : 94.36%
    F1-score       : 90.42%

The final application successfully performs:

    Chest X-ray Upload
            ↓
    AI Inference
            ↓
    Pneumonia Probability
            ↓
    NORMAL / PNEUMONIA
            ↓
    Confidence Display

---

## Final Project Status

    Dataset Pipeline         ✅
    Model Development        ✅
    Model Training           ✅
    Model Evaluation         ✅
    Single Image Inference   ✅
    Prediction Visualization ✅
    Streamlit UI             ✅
    GitHub Repository        ✅
    Cloud Deployment         ✅
    Cloud Inference Test     ✅
    Documentation            🔄

The core AI application is functionally complete and publicly deployed.

---

## Live Demo

**Streamlit Application:**

https://ai-medical-image-analysis-kxbuyy5vq8oxzks3jnr8gb.streamlit.app/

---

## Author

**Pranav**

AI / Machine Learning Engineering Project

Built as an industry-oriented proof-of-work demonstrating the practical application of deep learning, computer vision, medical image classification, application development, and cloud deployment.

---

## Medical Disclaimer

This project is intended strictly for educational, engineering, research, and portfolio purposes.

It is **not a medical device** and has not been clinically validated.

Predictions generated by this system must not be used as a substitute for professional medical diagnosis, treatment, or medical decision-making.

Any real-world medical deployment would require appropriate clinical validation, regulatory compliance, representative clinical data, patient-safety evaluation, qualified medical oversight, and appropriate deployment controls.