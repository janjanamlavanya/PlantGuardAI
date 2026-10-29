# 🌿 PlantGuardAI — Tomato Leaf Disease Detection Using Deep Learning

## Project Overview

PlantGuardAI is a deep learning application that classifies tomato leaf images into two categories: **Tomato Early Blight** and **Healthy Tomato Leaf**.

The project uses transfer learning with MobileNetV2 and provides a Streamlit web interface where users can upload leaf images and view model predictions and class probabilities.

## Key Features

- Tomato leaf image classification using deep learning.
- Transfer learning with pretrained MobileNetV2.
- Image resizing and data augmentation.
- Class-weighted training to address class imbalance.
- Training, validation, and independent test datasets.
- Model evaluation using accuracy, precision, recall, F1-score, and a confusion matrix.
- Interactive Streamlit web application.
- Prediction confidence and class probability visualization.

## Technologies Used

- Python
- TensorFlow and Keras
- MobileNetV2
- NumPy
- Scikit-learn
- Matplotlib
- Streamlit
- Pillow

## Dataset

The dataset contains 2,591 tomato leaf images across two classes.

| Dataset split | Early Blight | Healthy | Total |
|---|---:|---:|---:|
| Training | 700 | 1,113 | 1,813 |
| Validation | 150 | 238 | 388 |
| Testing | 150 | 240 | 390 |
| **Total** | **1,000** | **1,591** | **2,591** |

The images are divided into training, validation, and test sets using a reproducible random seed.

## Model Architecture

1. Input tomato leaf image.
2. Resize image to 160 × 160 pixels.
3. Apply data augmentation during training.
4. Apply MobileNetV2-compatible pixel scaling.
5. Extract features using pretrained MobileNetV2.
6. Apply global average pooling and dropout.
7. Classify into Early Blight or Healthy using a softmax output layer.

The model is trained in two phases: classification-head training and fine-tuning.

## Evaluation Results

The V4 model achieved the following results on the held-out test dataset:

- **Test accuracy:** 98.46%
- **Correct predictions:** 384 out of 390
- **Incorrect predictions:** 6 out of 390

### Test confusion matrix

| Actual class | Predicted Early Blight | Predicted Healthy |
|---|---:|---:|
| Early Blight | 144 | 6 |
| Healthy | 0 | 240 |

These results apply to the specific two-class test dataset. Performance on new images, other tomato diseases, and real-world field conditions may differ.

## Project Structure

```text
PlantGuardAI/
├── app.py
├── requirements.txt
├── README.md
├── data/
│   ├── raw/
│   ├── raw_clean/
│   └── processed_v3/
│       ├── train/
│       ├── val/
│       └── test/
├── models/
│   └── plantguard_mobilenetv2_v4.keras
├── results/
│   └── v4_test_confusion_matrix.png
├── src/
│   ├── train_model_v2.py
│   ├── evaluate_test_v4.py
│   ├── predict_image.py
│   └── split_dataset_clean.py
└── test_images/
    └── tomato_test.JPG
```

## Installation and Setup

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd PlantGuardAI
```

Replace `YOUR_GITHUB_REPOSITORY_URL` with your actual repository URL.

### 2. Create and activate a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
python -m pip install -r requirements.txt
```

### 4. Prepare the model and dataset

Ensure the trained model is available at:

```text
models/plantguard_mobilenetv2_v4.keras
```

The dataset is needed for training and evaluation. The Streamlit application requires the saved model.

### 5. Run the web application

```powershell
python -m streamlit run app.py
```

Upload a tomato leaf image and click **Analyze Leaf** to view the prediction.

## Model Evaluation

To evaluate the model on the held-out test dataset:

```powershell
python .\src\evaluate_test_v4.py
```

To predict an image using the command line:

```powershell
python .\src\predict_image.py
```

## Limitations

- The model recognizes only Tomato Early Blight and Healthy Tomato Leaf.
- It does not identify all plant diseases or other crop species.
- High confidence does not guarantee a correct prediction.
- Further validation on independent datasets and real-world field images is necessary before practical agricultural use.

## Future Enhancements

- Add more crops and disease categories.
- Explore Vision Transformers and segmentation models.
- Integrate explainable AI visualizations such as Grad-CAM.
- Add a disease severity estimation module.
- Improve evaluation using diverse, independently sourced images.
- Deploy the application online.

## Author

Developed as an AI/ML portfolio project demonstrating deep learning, transfer learning, model evaluation, and web application development.
