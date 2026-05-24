# Hotel Booking Cancellation Prediction

> A machine learning classification pipeline to predict hotel booking cancellations using supervised and unsupervised learning

![Python](https://img.shields.io/badge/Language-Python%203-blue)
![Platform](https://img.shields.io/badge/Platform-Google%20Colab-orange)
![Course](https://img.shields.io/badge/Course-CSE422%20Machine%20Learning-green)
![University](https://img.shields.io/badge/University-BRAC%20University-red)

---

## Overview

This project builds a full end-to-end machine learning pipeline on the **Hotel Bookings dataset** to predict whether a hotel booking will be cancelled or not (`is_canceled`). The pipeline covers data exploration, preprocessing, feature engineering, model training, and evaluation using multiple supervised classifiers and one unsupervised clustering method.

---

## Dataset

- **File:** `hotel_bookings.csv`
- **Target variable:** `is_canceled` (0 = Not Cancelled, 1 = Cancelled)
- **Size:** 119,390 rows × 32 features
- **Features:** Mix of numerical and categorical hotel booking attributes including lead time, arrival date, number of guests, room type, deposit type, and more

---

## Project Pipeline

### 1. Data Loading and Exploration
- Load dataset using pandas
- Inspect shape, data types, missing values
- Separate numerical and categorical features
- Summary statistics — mean, variance, skewness
- Class distribution analysis of target variable

### 2. Visualizations
- Histograms of all numerical features
- Bar charts for categorical feature distributions
- Density plots for numerical features
- Correlation heatmap (Pearson) for all numerical features
- Pearson correlation specifically with `is_canceled`
- Class imbalance bar chart

### 3. Data Preprocessing
- Drop `company` column (too many missing values)
- Fill `agent` missing values with 0
- Fill `children` missing values with 0
- Fill `country` missing values with mode
- Drop `reservation_status` and `reservation_status_date` to prevent data leakage
- Label encode all categorical columns
- StandardScaler normalization on all features

### 4. Train-Test Split
- 80/20 stratified split to preserve class balance
- `random_state=42` for reproducibility

### 5. Models Trained

| Model | Type |
|---|---|
| K-Nearest Neighbors (KNN) | Supervised — Classification |
| Logistic Regression | Supervised — Classification |
| Naive Bayes (GaussianNB) | Supervised — Classification |
| Neural Network (MLPClassifier) | Supervised — Classification |
| KMeans Clustering | Unsupervised — Clustering |

### 6. Evaluation Metrics
- Accuracy
- Precision
- Recall
- AUC (Area Under ROC Curve)
- Confusion Matrix for each model
- ROC Curve comparison across all models
- Elbow method for optimal KMeans cluster count

---

## Results Summary

All four supervised models were evaluated and compared across accuracy, precision, recall, and AUC. ROC curves and confusion matrices are generated for each model. Bar chart comparison of model accuracies is included.

---

## Libraries Used

```
pandas
numpy
matplotlib
seaborn
scikit-learn
  - KNeighborsClassifier
  - LogisticRegression
  - GaussianNB
  - MLPClassifier
  - KMeans
  - StandardScaler
  - LabelEncoder
  - train_test_split
  - accuracy_score, precision_score, recall_score
  - roc_auc_score, roc_curve
  - confusion_matrix
```

---

## How to Run

### On Google Colab (recommended)
1. Upload `hotel_bookings.csv` to `/content/` in your Colab environment
2. Open `01_22301142_22311003_Code.ipynb` in Google Colab
3. Run all cells in order from top to bottom

### Locally
1. Clone this repository
2. Install dependencies:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```
3. Place `hotel_bookings.csv` in the same directory as the script
4. Update the dataset path in the code:
```python
dataset = pd.read_csv("hotel_bookings.csv")
```
5. Run the script:
```bash
python 01_22301142_22311003_code.py
```

---

## File Structure

```
├── 01_22301142_22311003_Code.ipynb   # Main Colab notebook
├── 01_22301142_22311003_code.py      # Exported Python script
├── hotel_bookings.csv                # Dataset (not included — download separately)
└── README.md                         # This file
```

---

## Dataset Source

The hotel bookings dataset is publicly available on Kaggle:
[Hotel Booking Demand Dataset](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand)

---

## Project Context

This project was developed for **CSE422 — Machine Learning** at BRAC University as a full classification pipeline assignment covering supervised and unsupervised learning methods on a real-world imbalanced dataset.

---

## Team

**Student IDs:** 22301142, 22311003
**Course:** CSE422 — Machine Learning
**Institution:** BRAC University

---

## License

This project is open source and free to use for educational purposes.
