
````markdown
# Waste Type Classification using Supervised Learning

A comparative machine learning project for classifying waste images into multiple categories using classical supervised learning techniques and handcrafted feature extraction.

## Project Overview

This project focuses on **waste type classification** for smart waste management using **supervised learning**.  
The system classifies waste images into the following 6 categories:

- cardboard_paper
- plastic
- glass
- metal
- organic
- trash

Since deep learning models were not used, this project applies **image preprocessing** and **feature extraction** techniques to convert images into meaningful numerical representations before training machine learning models.

The project compares the performance of four supervised learning algorithms:

- Logistic Regression
- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM)
- Random Forest

---

## Objectives

- Build a waste classification system using supervised learning
- Apply preprocessing to standardize image data
- Extract meaningful handcrafted features from images
- Train and evaluate four different machine learning models
- Compare model performance and identify the best-performing classifier

---

## Dataset

The dataset was organized into **6 final classes**:

- `cardboard_paper`
- `plastic`
- `glass`
- `metal`
- `organic`
- `trash`

### Why classes were merged
Some original classes were visually similar, so they were merged to reduce inter-class confusion and improve classification performance.

Examples:
- `cardboard + paper -> cardboard_paper`
- `food organics + vegetation -> organic`
- `miscellaneous trash + textile trash -> trash`

---

## Project Workflow

```text
Raw Images
→ Preprocessing
→ Feature Extraction
→ Feature Scaling / Normalization / PCA
→ Model Training
→ Evaluation
→ Model Comparison
````

---

## Preprocessing

Preprocessing was applied to make the image data consistent and suitable for classical machine learning models.

### Preprocessing steps

* Resize images to a fixed size
* Convert images into a consistent format
* Organize dataset into final 6 classes
* Prepare train and test datasets

---

## Feature Extraction

Because classical ML models cannot directly understand raw images effectively, feature extraction was used to convert each image into a numerical feature vector.

### Extracted features

* **HSV Histogram** – captures color information
* **HOG (Histogram of Oriented Gradients)** – captures shape and edge patterns
* **LBP (Local Binary Pattern)** – captures texture information
* **Edge Feature** – captures edge density information

### Additional processing

* Standard Scaling
* L2 Normalization
* PCA (Principal Component Analysis)

---

## Models Used

### 1. Logistic Regression

A linear baseline classifier used for comparison.

### 2. K-Nearest Neighbors (KNN)

A distance-based classifier used to identify similar waste samples.

### 3. Support Vector Machine (SVM)

A powerful classifier for high-dimensional feature spaces.

### 4. Random Forest

An ensemble learning model using multiple decision trees.

---

## Project Structure

```text
Waste-Classification-Project/
│
├── dataset_6/
│   ├── cardboard_paper/
│   ├── plastic/
│   ├── glass/
│   ├── metal/
│   ├── organic/
│   └── trash/
│
├── output_v2/
│   ├── logistic_regression/
│   ├── knn/
│   ├── svm/
│   └── random_forest/
│
├── src/
│   ├── preprocess_v2.py
│   ├── train_logistic_regression.py
│   ├── train_knn.py
│   ├── train_svm.py
│   ├── train_random_forest.py
│   └── compare_models.py
│
├── requirements.txt
└── README.md
```

---

## Installation

### 1. Clone the repository

```bash
git clone <your-github-repo-link>
cd <your-project-folder>
```

### 2. Create virtual environment

```bash
python -m venv venv
```

### 3. Activate virtual environment

#### Windows

```bash
venv\Scripts\activate
```

#### Mac/Linux

```bash
source venv/bin/activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Requirements

Example `requirements.txt`:

```txt
numpy
pandas
opencv-python
scikit-learn
scikit-image
matplotlib
Pillow
joblib
```

---

## How to Run

### Step 1 – Preprocess the dataset

```bash
cd src
python preprocess_v2.py
```

### Step 2 – Train models

#### Logistic Regression

```bash
python train_logistic_regression.py
```

#### KNN

```bash
python train_knn.py
```

#### SVM

```bash
python train_svm.py
```

#### Random Forest

```bash
python train_random_forest.py
```

### Step 3 – Compare models

If you have a comparison script:

```bash
python compare_models.py
```

---

## Outputs

For each model, the system saves:

* `model.pkl`
* `metrics.csv`
* `classification_report.txt`
* `confusion_matrix.png`
* `overall_metrics.png`
* `per_class_f1.png`

Example:

```text
output_v2/
└── svm/
    ├── model.pkl
    ├── metrics.csv
    ├── classification_report.txt
    ├── confusion_matrix.png
    ├── overall_metrics.png
    └── per_class_f1.png
```

---

## Evaluation Metrics

The following metrics were used to evaluate model performance:

* Accuracy
* Precision
* Recall
* F1-Score
* Confusion Matrix
* Per-Class F1 Score

---

## Best Model

According to the experimental results, **SVM** achieved the best overall performance among the four tested models.

### Final ranking

1. SVM
2. Random Forest
3. Logistic Regression
4. KNN

---

## Why SVM Performed Best

SVM performed best because it handled high-dimensional handcrafted features more effectively than the other classifiers. It was able to separate the waste categories with stronger generalization and better balance across classes.

---

## Why Preprocessing and Feature Extraction Were Necessary

Preprocessing was used to standardize the images and prepare them in a consistent format.
Feature extraction was necessary because classical machine learning models cannot directly interpret raw image data efficiently.

By extracting color, texture, shape, and edge-based features, the models were able to learn meaningful patterns for waste classification.

---

## Applications

This project can be useful for:

* Smart waste sorting systems
* Recycling automation
* AI-based waste management solutions
* Educational supervised learning projects

---

## Future Improvements

* Use a larger dataset
* Improve class balance
* Add more advanced feature engineering
* Try deep learning approaches in future work
* Build a real-time web or mobile application for waste prediction

---

## Team / Group Members

Add your group member details here:

* Member 1 – Logistic Regression
* Member 2 – KNN
* Member 3 – SVM
* Member 4 – Random Forest

Example:

* IT – Member Name – Logistic Regression
* IT – Member Name – KNN
* IT – Member Name – SVM
* IT – Member Name – Random Forest

---

## Conclusion

This project successfully demonstrated how supervised learning can be applied to waste type classification using handcrafted image features. Among the evaluated models, SVM achieved the best performance, showing that classical machine learning methods can still produce strong results when supported by effective preprocessing and feature extraction.

---

## License

This project is for academic and educational purposes.
