# 🧠 E-Commerce Decision Intelligence Platform

A machine learning project that predicts whether an online shopper will make a purchase and estimates their spending amount, using multiple ML models, SHAP explainability, customer persona mapping, and a Streamlit web dashboard.

## Overview

This project uses the [Online Shoppers Intention dataset](https://archive.ics.uci.edu/ml/datasets/Online+Shoppers+Purchasing+Intention+Dataset) for binary classification — predicting the `Revenue` column (True/False → 1/0) — and regression to estimate a synthesized `SpendingAmount` proxy derived from `PageValues`.

## Project Structure

```
Purchase_predictor/
├── data/
│   └── online_shoppers_intention.csv   # Dataset
├── results/
│   ├── metrics.txt                     # Saved evaluation metrics
│   └── predictions.csv                 # Model predictions output
├── src/
│   ├── preprocessing.py                # Data loading, scaling, encoding, train/test split
│   ├── feature_engineering.py          # SelectKBest, PCA, Factor Analysis, persona labeling
│   ├── modeling.py                     # Decision Tree, Naive Bayes, Linear Regression
│   ├── evaluate.py                     # Classification report & confusion matrix
│   ├── explainability.py               # SHAP TreeExplainer & LinearExplainer
│   ├── simulation.py                   # Scenario simulation engine
│   └── utils.py                        # PCA scatter plot utility
├── app.py                              # Streamlit web dashboard
├── main.py                             # CLI entry point
├── predictions.csv                     # Root-level predictions output
└── requirements.txt
```

## ML Pipeline

```
Raw CSV → Preprocess → Feature Selection → Dimensionality Reduction → Models → Evaluate → Explain
```

1. **Preprocess** (`preprocessing.py`) — StandardScaler on numerical columns, OneHotEncoder on categorical/boolean columns, 80/20 train-test split. Synthesizes a `SpendingAmount` regression target from `PageValues`.
2. **Feature Selection** (`feature_engineering.py`) — Top 15 features via `SelectKBest` with mutual information scoring.
3. **Dimensionality Reduction** (`feature_engineering.py`) — PCA (5 components) and Factor Analysis (3 components); models train on selected features.
4. **Persona Mapping** (`feature_engineering.py`) — Factor Analysis scores mapped to human-readable labels: *High Intent Buyer*, *Price Sensitive / Cart Abandoner*, *Casual Browser*.

## Models

| Model | Algorithm | Task | Metric |
|---|---|---|---|
| Decision Tree | DecisionTreeClassifier (max_depth=6) | Classification | Accuracy |
| Naive Bayes | GaussianNB | Classification | Accuracy |
| Linear Regression | LinearRegression | Regression (SpendingAmount) | RMSE |

> ⚠️ No Neural Network is implemented. The `requirements.txt` lists `tensorflow`/`keras` but they are unused.

## Explainability (SHAP)

- **Decision Tree** — `shap.TreeExplainer` for global feature importance
- **Linear Regression** — `shap.LinearExplainer` for revenue driver analysis
- Both exposed as summary plots and local waterfall plots in the dashboard

## Scenario Simulator

`simulation.py` allows multiplying key features (e.g. `PageValues`, `BounceRates`, `Informational_Duration`) on a base user row and re-running predictions to show delta in purchase probability and predicted spend.

## Installation

```bash
pip install -r requirements.txt
```

> Note: Remove `tensorflow` and `keras` from `requirements.txt` if not needed — they are not used by any module.

## Usage

**CLI mode** — trains all models, prints metrics, generates business insights:
```bash
python main.py
```

**Web dashboard** — 4-tab interactive Streamlit app:
```bash
streamlit run app.py
```

### Dashboard Tabs

| Tab | Content |
|---|---|
| 1️⃣ Purchase Classifier & Explainability | Accuracy, confusion matrix, ROC curve, PR curve, SHAP summary |
| 2️⃣ Revenue Predictor | Regression RMSE, SHAP global importance for spend drivers |
| 3️⃣ Customer Personas | Factor Analysis scatter, persona labels with explanations |
| 4️⃣ Scenario Simulator | Sliders to adjust features, delta metrics for prob & spend |

## Output

- `predictions.csv` — True labels vs predictions from all models
- `results/metrics.txt` — Saved evaluation metrics

## Requirements

- Python 3.8+
- pandas, numpy, scikit-learn, matplotlib, seaborn, streamlit, shap
