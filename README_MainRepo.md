# 🤖 DevelopersHub Corporation — AI/ML Engineering Internship

> A portfolio of completed machine learning projects spanning data exploration, predictive modeling, and binary classification.

![Projects Completed](https://img.shields.io/badge/Projects%20Completed-4%20%2F%206-4ade80?style=flat-square&labelColor=0b0d11)
![Language](https://img.shields.io/badge/Language-Python%203.10-5b8fff?style=flat-square&labelColor=0b0d11)
![Framework](https://img.shields.io/badge/Framework-scikit--learn-a78bfa?style=flat-square&labelColor=0b0d11)
![Due Date](https://img.shields.io/badge/Due%20Date-3%20April%202026-f0c060?style=flat-square&labelColor=0b0d11)

---

## 📋 Project Overview

| # | Project Title | Dataset | Model(s) | Status |
|---|---------------|---------|----------|--------|
| 01 | Dataset Exploration & Visualization — Iris | Iris (UCI) | EDA only | ✅ Complete |
| 02 | Short-Term Stock Price Prediction — AAPL | Yahoo Finance via yfinance | Linear Regression | ✅ Complete |
| 03 | Heart Disease Risk Prediction — UCI Cleveland | UCI Cleveland Heart Disease | Logistic Regression · Decision Tree | ✅ Complete |
| 06 | House Price Prediction — Regression & Feature Analysis | Synthetic / Kaggle-structured | Linear Regression · Gradient Boosting | ✅ Complete |

---

## 📁 Project Structure

```
developershub-aiml-internship/
├── Task_1.ipynb               # Iris Dataset — EDA & Visualization
├── Task_2.ipynb               # AAPL Stock Price Prediction
├── Task_3.ipynb               # Heart Disease Risk Classification
├── Task_6.ipynb               # House Price Regression
├── README.md                  # This file
└── README.html                # Styled portfolio version
```

---

## 🔬 Project 01 — Dataset Exploration & Visualization

**Objective:** Load, inspect, and visualize the Iris dataset to understand data distributions and feature relationships.

**Dataset:** Iris Dataset — UCI Repository (loaded via `seaborn`)

**What was done:**
- Loaded dataset using `pandas` — printed shape, columns, `.head()`, `.info()`, and `.describe()`
- Scatter plot of sepal length vs sepal width colored by species, plus a full pairplot across all features
- Histograms for all 4 features with KDE overlays grouped by species
- Box plots per feature to identify outliers across all three species

**Key Findings:**
- Petal length and petal width are the most discriminating features between species
- Iris Setosa is clearly linearly separable from the other two species
- Versicolor and Virginica overlap in sepal dimensions but are separable via petal measurements

**File:** `Task_1.ipynb`

---

## 📈 Project 02 — Short-Term Stock Price Prediction

**Objective:** Use historical Apple (AAPL) stock data to predict the next day's closing price.

**Dataset:** Apple Inc. (AAPL) — Yahoo Finance via `yfinance` (January 2020 – January 2024)

**Model:** Linear Regression

| Metric | Value |
|--------|-------|
| Features used | Open, High, Low, Volume |
| Target | Next-day Close price |
| Split strategy | `shuffle=False` — time series order preserved |
| Evaluation | MSE · RMSE |

**Key Findings:**
- Linear Regression captures the general price trend well for next-day prediction
- Using `shuffle=False` is critical for time series data — prevents leakage from the future
- Today's Open/High/Low values are strong proxies for tomorrow's Close price

**File:** `Task_2.ipynb`

---

## 🫀 Project 03 — Heart Disease Risk Prediction

**Objective:** Predict whether a patient is at risk of heart disease using 13 clinical health features.

**Dataset:** UCI Heart Disease Dataset — Cleveland subset (242 clean records, 13 features, binary target)

**Model Results:**

| Model | Accuracy | ROC-AUC |
|-------|----------|---------|
| **Logistic Regression** | **93.9%** | **0.982** |
| Decision Tree (max_depth=5) | 91.8% | 0.942 |

**What was done:**
- Loaded UCI Cleveland dataset and handled `?` missing values in `ca` and `thal` columns
- Binarised target: `0` = no disease, `1` = disease present
- Full EDA — class distribution, age histograms, max heart rate scatter, chest pain type analysis, cholesterol box plots
- Trained both models; evaluated with accuracy, confusion matrix, ROC curve, and AUC score
- Feature importance (Decision Tree) and coefficient magnitude (Logistic Regression) charts

**Key Findings:**
- `ca` (number of major vessels blocked) and `cp` (chest pain type) are the top two predictors
- `oldpeak` (ST depression) and `thal` (thalassemia type) are the next most important features
- Logistic Regression significantly outperforms Decision Tree on AUC — 0.982 vs 0.942
- Model achieves strong clinical screening performance, suitable as a patient triage support tool

**File:** `Task_3.ipynb`

---

## 🏠 Project 06 — House Price Prediction

**Objective:** Predict house prices using property features — square footage, bedrooms, location, age, and amenities.

**Dataset:** Synthetic dataset (1,500 records, 9 features) structured to mirror the Kaggle House Price dataset

**Model Results:**

| Model | MAE | RMSE |
|-------|-----|------|
| Linear Regression | $33,665 | $41,341 |
| **Gradient Boosting (200 estimators)** | **$24,374** | **$30,626** |

**What was done:**
- Generated a realistic dataset with location-based price premiums and natural Gaussian noise
- Preprocessing — Label Encoding for `Location`, StandardScaler applied for Linear Regression
- Full EDA — price distribution, price vs SqFt scatter by location, average price by location, correlation heatmap, box plots by bedrooms and garage spaces
- Trained both models; plotted actual vs predicted scatter with a perfect-fit reference line
- Feature importance bar chart from Gradient Boosting regressor

**Key Findings:**
- `SqFt` (square footage) is by far the strongest single predictor of house price
- Location is the second most important feature — Downtown/Uptown carry a ~$80,000 premium
- Gradient Boosting outperforms Linear Regression by 27% on MAE, capturing non-linear interactions
- `Age` has a consistent negative effect; `Pool` and `Garage` add measurable value premiums

**File:** `Task_6.ipynb`

---

## ⚙️ Setup & Installation

```bash
# Core dependencies — all projects
pip install pandas numpy matplotlib seaborn scikit-learn

# Project 02 — Stock data fetching
pip install yfinance
```

---

## 🧠 Skills Demonstrated

| Category | Skills |
|----------|--------|
| **Data Engineering** | Data loading, cleaning, preprocessing, missing value handling |
| **Visualization** | Matplotlib, Seaborn — scatter plots, histograms, box plots, heatmaps |
| **Regression** | Linear Regression, Gradient Boosting Regressor |
| **Classification** | Logistic Regression, Decision Tree Classifier |
| **Model Evaluation** | MAE, RMSE, Accuracy, ROC-AUC, Confusion Matrix, Feature Importance |
| **Time Series** | Sequential train/test splits, preventing data leakage |

---

## 🔗 Related Repository

> Task 04 — General Health Query Chatbot (Prompt Engineering & LLM) is maintained in a separate repository.
> 👉 [developershub-health-chatbot](https://github.com/ArmanAdilMangat/developershub-health-chatbot)

---

## 👤 Author

**Arman Adil Mangat**
AI/ML Engineering Intern — DevelopersHub Corporation
