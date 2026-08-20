# AI Based Medical Image Analysis – Chest X-Ray Pneumonia Detection

An AI-based medical image classification system that analyzes chest X-ray images and predicts whether an image represents **NORMAL** or **PNEUMONIA** using MobileNetV2 transfer learning.

> **Medical Disclaimer:** This project is developed for educational, engineering, research, and portfolio purposes. It is not intended for clinical diagnosis, medical decision-making, or replacement of qualified medical professionals.

---

## 1. Project Overview

This project implements a complete deep-learning pipeline for binary classification of chest X-ray images.

The system takes a chest X-ray image as input and classifies it into one of two classes:

- **NORMAL**
- **PNEUMONIA**

The project demonstrates an end-to-end computer vision and machine-learning workflow:

```text
Chest X-ray Dataset
        ↓
Data Loading
        ↓
Image Preprocessing
        ↓
MobileNetV2 Transfer Learning
        ↓
Model Training
        ↓
Model Evaluation
        ↓
Prediction Visualization
        ↓
Single Image Inference
```

The project is designed as an industry-oriented proof-of-work demonstrating practical skills in:

- Computer Vision
- Deep Learning
- Transfer Learning
- Medical Image Classification
- Python
- TensorFlow / Keras
- Model Evaluation
- Data Visualization
- Machine Learning Project Development

---

## 2. Problem Statement

Chest X-ray images contain visual information that can be analyzed using computer vision and deep-learning techniques.

The objective of this project is to develop a machine-learning system that learns visual patterns from labeled chest X-ray images and classifies them as either:

```text
NORMAL
```

or:

```text
PNEUMONIA
```

The project focuses on building a practical AI pipeline using a public chest X-ray dataset.

It is not intended to replace professional medical diagnosis.

---

## 3. Project Objectives

The main objectives are to:

- Build a reproducible medical image classification pipeline.
- Load and preprocess chest X-ray images.
- Normalize image data for deep-learning models.
- Apply transfer learning using MobileNetV2.
- Handle class imbalance using class weights.
- Train and validate the classification model.
- Evaluate the trained model on a separate test set.
- Generate useful training and prediction outputs.
- Implement single-image inference.
- Document the complete engineering workflow for GitHub.

---

## 4. Industry Relevance

Medical imaging is an important application area for artificial intelligence and computer vision.

AI-based image analysis systems can assist healthcare workflows by processing large collections of medical images and identifying visual patterns that may require further professional review.

Potential industry environments include:

- Hospitals
- Diagnostic laboratories
- Radiology centers
- Medical imaging companies
- Health-tech companies
- Medical AI research organizations

This project demonstrates the engineering workflow behind such systems using public data rather than real hospital infrastructure.

The implementation is intended as a technical proof-of-work and not as a clinically validated medical product.

---

## 5. Tech Stack

### Programming Language

- Python

### Deep Learning

- TensorFlow
- Keras
- MobileNetV2

### Computer Vision and Image Processing

- OpenCV
- Pillow

### Numerical Computing

- NumPy

### Model Evaluation

- Scikit-learn

### Visualization

- Matplotlib

### Development Environment

- Visual Studio Code
- Python virtual environment
- Git / GitHub

---

## 6. Dataset

The project uses a public chest X-ray dataset containing two classes:

- **NORMAL**
- **PNEUMONIA**

The verified dataset contains:

| Class | Images |
|---|---:|
| NORMAL | 1,341 |
| PNEUMONIA | 3,875 |
| **Total** | **5,856** |

The dataset is organized into:

```text
train/
val/
test/
```

The verified image distribution is:

| Dataset | Images |
|---|---:|
| Training | 5,216 |
| Validation | 16 |
| Test | 624 |
| **Total** | **5,856** |

The project uses the existing train, validation, and test directories rather than creating a new split inside the code.

---

## 7. Dataset Structure

The expected dataset structure is:

```text
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
```

The dataset path is configured in:

```text
src/data_pipeline.py
```

The local dataset path is defined through the `DATASET_DIR` variable.

Because the dataset is stored locally, the dataset itself is not included in the GitHub repository.

---

## 8. System Architecture

The complete system follows this architecture:

```text
                    CHEST X-RAY DATASET
                            │
                            ▼
                  DATASET LOADING
                            │
                            ▼
                  IMAGE PREPROCESSING
                  ┌─────────┴─────────┐
                  │                   │
              Resize              RGB Input
            224 × 224              3 Channels
                  │                   │
                  └─────────┬─────────┘
                            ▼
                    NORMALIZATION
                       0–255 → 0–1
                            │
                            ▼
                 MOBILENETV2 BACKBONE
                  ImageNet Pretrained
                   Frozen Feature
                     Extractor
                            │
                            ▼
                GLOBAL AVERAGE POOLING
                            │
                            ▼
                     DROPOUT 0.30
                            │
                            ▼
                    DENSE 1 UNIT
                            │
                            ▼
                   SIGMOID OUTPUT
                            │
                            ▼
                 PNEUMONIA PROBABILITY
                            │
                            ▼
                     0.5 THRESHOLD
                      /           \
                     /             \
                NORMAL          PNEUMONIA
```

---

## 9. Data Pipeline

The data pipeline is implemented in:

```text
src/data_pipeline.py
```

TensorFlow's `image_dataset_from_directory()` is used to load the images.

### Configuration

| Parameter | Value |
|---|---|
| Label mode | Binary |
| Color mode | RGB |
| Image size | 224 × 224 |
| Channels | 3 |
| Batch size | 32 |
| Random seed | 42 |
| Training shuffle | Enabled |
| Validation shuffle | Disabled |
| Test shuffle | Disabled |

### Preprocessing

Each image is:

1. Loaded as RGB.
2. Resized to `224 × 224`.
3. Normalized from pixel range `0–255` to `0–1`.
4. Passed through the TensorFlow data pipeline.
5. Prefetched using `tf.data.AUTOTUNE`.

The preprocessing flow is:

```text
Original Chest X-ray
        ↓
Resize to 224 × 224
        ↓
RGB / 3 Channels
        ↓
Pixel Normalization
0–255 → 0–1
        ↓
Model Input
```

No image augmentation is implemented in the current version.

---

## 10. Model Architecture

The model is implemented in:

```text
src/model.py
```

The project uses **MobileNetV2 transfer learning**.

MobileNetV2 is initialized using ImageNet pretrained weights and is used as a frozen feature extractor.

### Base Model

```text
MobileNetV2
├── ImageNet pretrained weights
├── include_top = False
└── Frozen feature extractor
```

The pretrained MobileNetV2 layers remain frozen during the current training process.

### Classification Head

The classification head is:

```text
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
Pneumonia Probability
```

### Model Input

```text
224 × 224 × 3
```

### Model Output

```text
1 sigmoid probability
```

The sigmoid output represents the predicted probability of the **PNEUMONIA** class.

---

## 11. MobileNetV2 Input Scaling

The data pipeline produces normalized pixel values in the range:

```text
0 to 1
```

The model applies an additional scaling operation before passing the images into MobileNetV2:

```text
0–1
 ↓
-1–1
```

This transformation is implemented using a Keras `Rescaling` layer inside the model.

Keeping the transformation inside the model helps maintain consistent preprocessing during both training and inference.

---

## 12. Training Methodology

Training is implemented in:

```text
src/train.py
```

The model uses the following configuration:

| Parameter | Value |
|---|---|
| Architecture | MobileNetV2 Transfer Learning |
| Input size | 224 × 224 × 3 |
| Optimizer | Adam |
| Learning rate | 0.0001 |
| Loss function | Binary Crossentropy |
| Epochs | 5 |
| Batch size | 32 |
| Dropout | 0.30 |
| Base model | Frozen |

### Class Weighting

The dataset contains more PNEUMONIA images than NORMAL images.

To account for this imbalance, class weights are calculated from the training class counts and passed to the model during training.

The training class counts used by the implementation are:

```text
NORMAL     : 1,341
PNEUMONIA  : 3,875
```

The class weights are passed through the Keras training process using:

```python
class_weight=class_weights
```

This gives the minority NORMAL class greater influence during training.

---

## 13. Training Callbacks

Two callbacks are used during training.

### Early Stopping

The model monitors:

```text
val_loss
```

with:

```text
patience = 2
restore_best_weights = True
```

This allows training to stop when validation loss stops improving while restoring the best weights observed during training.

### Model Checkpoint

The best model according to validation loss is saved using:

```text
models/chest_xray_mobilenetv2.keras
```

The checkpoint uses:

```text
save_best_only = True
```

This ensures that the best validation-loss model is retained rather than simply the final epoch model.

---

## 14. Training History and Generated Outputs

Training history is saved to:

```text
outputs/training_history.csv
```

The project also generates training-related visualization output during the visualization stage.

Useful project evidence includes:

- Training logs
- Training accuracy
- Validation accuracy
- Training history
- Prediction results
- Preprocessing visualization

The generated outputs are kept outside the core source-code modules and are excluded from Git tracking according to the current `.gitignore` configuration.

## 15. Model Evaluation

Model evaluation is implemented in:

```text
src/evaluate.py
```

The saved model is evaluated against the test dataset containing:

```text
624 images
```

The evaluation calculates:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
- Classification Report

### Classification Threshold

The model produces a sigmoid probability.

The project uses:

```text
0.5
```

as the classification threshold.

Therefore:

```text
Probability >= 0.5
        ↓
PNEUMONIA
```

and:

```text
Probability < 0.5
        ↓
NORMAL
```

---

## 16. Test Results

The verified test-set results are:

| Metric | Result |
|---|---:|
| Test Images | **624** |
| Accuracy | **87.57%** |
| Precision | **86.79%** |
| Recall | **94.36%** |
| F1-score | **90.42%** |

### Confusion Matrix

```text
                 Predicted
              NORMAL  PNEUMONIA

Actual NORMAL    178      56
Actual PNEUMONIA  22     368
```

The model correctly classified:

- 178 NORMAL images
- 368 PNEUMONIA images

The model identified 368 out of 390 pneumonia cases in the test set.

The results show a relatively high recall for the PNEUMONIA class, which is particularly relevant when considering the importance of reducing missed positive cases in an educational medical-image classification setting.

These results should not be interpreted as clinical performance.

---

## 17. Single Image Inference

Single-image inference is implemented in:

```text
src/predict.py
```

The inference workflow is:

```text
Input X-ray
      ↓
Open Image
      ↓
Convert to RGB
      ↓
Resize to 224 × 224
      ↓
Normalize 0–255 → 0–1
      ↓
Load Trained Model
      ↓
Generate Sigmoid Probability
      ↓
Apply 0.5 Threshold
      ↓
NORMAL / PNEUMONIA
      ↓
Confidence + Probability
```

The prediction script reports:

- Input image path
- Predicted class
- Confidence
- PNEUMONIA probability

### Example Verified Prediction

One verified test-image inference produced:

```text
Prediction  : NORMAL
Confidence  : 0.6413
Probability : 0.3587
```

The probability represents the predicted probability of the PNEUMONIA class.

---

## 18. Project Outputs and Proof Assets

The project is designed to produce practical proof of each major stage.

Recommended proof assets include:

### Data Pipeline Proof

- Dataset verification output
- Image shape verification
- Pixel-range verification
- Class-name verification

### Model Proof

- Model summary
- Input shape
- Output shape
- MobileNetV2 architecture
- Frozen base-model confirmation

### Training Proof

- Training terminal output
- Epoch results
- Validation metrics
- Model checkpoint confirmation
- Training history

### Evaluation Proof

- Test-set metrics
- Confusion matrix
- Classification report

### Prediction Proof

- Test prediction visualization
- Single-image prediction output
- Predicted class and probability

### Documentation Proof

- System workflow diagram
- GitHub repository
- README preview

These assets should represent real outputs from the implemented project rather than decorative mockups.

---

## 19. Project Structure

The core repository structure is:

```text
Project-4-AI-Medical-Image-Analysis/
│
├── src/
│   ├── data_pipeline.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
│
├── .gitignore
├── requirements.txt
└── README.md
```

The project may also contain generated output and model directories locally.

The dataset itself is stored outside the repository.

The current `.gitignore` excludes:

- Virtual environment files
- Python cache files
- Raw and processed datasets
- Trained model files
- Generated outputs
- Environment files
- IDE-specific files

---

## 20. Installation

### Windows

Create a virtual environment:

```powershell
python -m venv .venv
```

Activate the environment:

```powershell
.venv\Scripts\Activate.ps1
```

Install the required libraries:

```powershell
pip install -r requirements.txt
```

### macOS / Linux

Create a virtual environment:

```bash
python3 -m venv .venv
```

Activate the environment:

```bash
source .venv/bin/activate
```

Install the required libraries:

```bash
pip install -r requirements.txt
```

---

## 21. Dataset Configuration

The dataset is not included in the GitHub repository.

After downloading and extracting the dataset, use the following structure:

```text
chest_xray/
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
```

Open:

```text
src/data_pipeline.py
```

and update:

```python
DATASET_DIR = Path(
    r"YOUR_LOCAL_CHEST_XRAY_DATASET_PATH"
)
```

Replace the placeholder with the local path of the `chest_xray` directory.

---

## 22. How to Run the Project

### Step 1 — Verify the Data Pipeline

Run:

```powershell
python src/data_pipeline.py
```

This verifies:

- Dataset loading
- Class names
- Image shapes
- Label shapes
- Pixel range
- Expected image format

### Step 2 — Verify the Model

Run:

```powershell
python src/model.py
```

This verifies:

- Model architecture
- Input shape
- Output shape
- MobileNetV2 configuration
- Frozen base model
- Model summary

### Step 3 — Train the Model

Run:

```powershell
python src/train.py
```

This performs:

- Dataset loading
- Class-weight calculation
- Model construction
- Model training
- Early stopping
- Best-model checkpointing
- Training-history saving

### Step 4 — Evaluate the Model

Run:

```powershell
python src/evaluate.py
```

This produces:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix
- Classification report

### Step 5 — Run Single Image Prediction

Run:

```powershell
python src/predict.py "<PATH_TO_XRAY_IMAGE>"
```

Example:

```powershell
python src/predict.py "D:\Datasets\Medical-Datasets\chest_xray\chest_xray\test\NORMAL\IM-0001-0001.jpeg"
```

The prediction script returns the predicted class, confidence, and PNEUMONIA probability.

---

## 23. Complete Workflow

The complete project execution flow is:

```text
1. Configure Environment
        ↓
2. Configure Dataset Path
        ↓
3. Verify Data Pipeline
        ↓
4. Verify Model Architecture
        ↓
5. Train MobileNetV2 Classifier
        ↓
6. Save Best Model
        ↓
7. Save Training History
        ↓
8. Evaluate on Test Dataset
        ↓
9. Analyze Metrics and Confusion Matrix
        ↓
10. Run Single Image Prediction
        ↓
11. Capture Project Evidence
        ↓
12. Document and Publish on GitHub
```

This workflow demonstrates a complete practical machine-learning development cycle from dataset handling through model evaluation and inference.

---

## 24. Limitations

This project is an educational and engineering demonstration rather than a clinical diagnostic system.

### Dataset Dependency

The model is trained on a specific public chest X-ray dataset. Performance may differ when applied to images from other datasets, hospitals, imaging devices, or patient populations.

### Validation Dataset

The provided dataset contains a relatively small validation set compared with the training and test sets.

### Frozen Feature Extractor

The MobileNetV2 feature extractor remains frozen during the current training process. The project does not perform fine-tuning of the pretrained layers.

### Binary Classification

The current system only predicts:

```text
NORMAL
```

or:

```text
PNEUMONIA
```

It does not diagnose other diseases or classify specific pneumonia subtypes.

### No Clinical Validation

The model has not been clinically validated and must not be used for medical diagnosis or treatment decisions.

### Dataset Generalization

Performance measured on this dataset does not guarantee equivalent performance on external clinical datasets.

---

## 25. Future Improvements

Possible future extensions include:

- Fine-tuning selected MobileNetV2 layers.
- Increasing the validation dataset size.
- Introducing controlled image augmentation.
- Testing additional transfer-learning architectures.
- Performing systematic hyperparameter tuning.
- Testing on an independent external dataset.
- Adding Grad-CAM or another explainability method.
- Building a local web-based inference interface.
- Adding experiment tracking and model versioning.
- Packaging the model for a controlled deployment environment.

These are future possibilities and are not part of the current implementation.

---

## 26. Learning Outcomes

This project provides practical experience in:

- Medical image classification
- Computer vision
- TensorFlow and Keras
- Transfer learning
- MobileNetV2
- Image preprocessing
- Binary classification
- Class imbalance handling
- Model training
- Early stopping
- Model checkpointing
- Classification metrics
- Confusion matrix analysis
- Prediction visualization
- Single-image inference
- Python project organization
- GitHub project development

---

## 27. Industry-Oriented Workflow

The project follows a practical machine-learning engineering workflow:

```text
Problem
   ↓
Public Dataset
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
Prediction
   ↓
Visualization
   ↓
Documentation
   ↓
GitHub Proof of Work
```

The objective is not to claim clinical deployment, but to demonstrate the engineering process required to develop and evaluate an AI-based medical image analysis system.

---

## 28. Project Showcase

The strongest project evidence should include:

- System workflow diagram
- Dataset preprocessing output
- Model architecture verification
- Training terminal output
- Training history visualization
- Test-set evaluation output
- Confusion matrix
- Test prediction visualization
- Single-image inference output
- GitHub repository
- README preview

The showcase should focus on real outputs generated by the working system.

This makes the repository easier for recruiters and technical reviewers to understand and verify.

---

## 29. Medical Disclaimer

This project is intended strictly for educational, engineering, research, and portfolio purposes.

It is **not a medical device** and has not been clinically validated.

The predictions generated by the model must not be used as a substitute for diagnosis, treatment, or professional medical judgment.

Any real-world medical application would require appropriate clinical validation, regulatory compliance, patient-safety evaluation, representative clinical data, qualified medical oversight, and appropriate deployment controls.

---

## 30. Author

**Pranav**

AI / Machine Learning Engineering Project

Built as an industry-oriented proof-of-work project demonstrating the practical application of deep learning and computer vision to medical image analysis.