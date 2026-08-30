# SDSS Stellar Classification — Kaggle Stacking Pipeline

A machine learning pipeline that classifies celestial objects from Sloan Digital Sky Survey (SDSS) photometric data using a regularized stacking ensemble of gradient-boosted models. Built for the Kaggle Playground Series — Season 6, Episode 6 ("Predicting Stellar Class") competition.

**Final accuracy: 96.3%** on an imbalanced multi-class target.

---

## Overview

The task is to classify astronomical objects observed by SDSS into their correct stellar class based on photometric and spectroscopic features. The dataset is tabular but domain-specific, with meaningful physical structure (celestial coordinates, redshift, photometric bands) and a non-trivial class imbalance across target categories.

This repo contains the full pipeline: exploratory data analysis, feature engineering, base model training, hyperparameter optimization, and the final stacked ensemble used for the leaderboard submission.

## Approach

**1. Exploratory Data Analysis**
Investigated feature distributions, class balance, and correlations across photometric bands and spectroscopic features to guide feature engineering decisions.

**2. Feature Engineering**
- Cyclical trigonometric encoding (sine/cosine transforms) for celestial coordinate features (RA/Dec), preserving angular continuity that raw degree values lose at wrap-around boundaries.
- Multi-class target encoding for categorical features, applied within a cross-validated loop to avoid target leakage.
- Additional domain-informed derived features from raw photometric measurements.

**3. Base Models**
Three gradient-boosted tree models were trained as base learners:
- **LightGBM**
- **CatBoost**
- **XGBoost**

Each was tuned independently before being combined.

**4. Hyperparameter Optimization**
Used **Optuna** (GPU-accelerated) to search the hyperparameter space for each base model, wrapped inside a **5-fold out-of-fold (OOF) cross-validation** loop. This keeps the tuning objective aligned with generalization performance rather than in-sample fit, and produces clean OOF predictions for the stacking stage.

**5. Stacking Ensemble**
Out-of-fold predictions from the three base models were used as meta-features for a **stacking classifier with an L2-regularized meta-learner**. The L2 penalty was chosen specifically to control overfitting at the meta-learner stage, given the correlated nature of the three base models' predictions.

## Results

| Metric | Score |
|---|---|
| Accuracy (final ensemble) | **96.3%** |
| Validation strategy | 5-fold OOF cross-validation |

## Repository Structure

```
.
├── notebooks/
│   └── stellar_classification.ipynb   # Full pipeline: EDA → features → training → stacking
├── src/
│   ├── features.py                    # Feature engineering (cyclical encoding, target encoding)
│   ├── train_base_models.py           # LightGBM / CatBoost / XGBoost training + Optuna tuning
│   └── stack.py                       # OOF meta-feature generation + stacking classifier
├── models/                            # Saved base model artifacts
├── submission.csv                     # Final Kaggle submission
├── requirements.txt
└── README.md
```

*(Adjust the tree above to match your actual file layout once everything is pushed from Colab.)*

## Tech Stack

Python · LightGBM · CatBoost · XGBoost · Optuna · Scikit-learn · Pandas · NumPy

## Setup & Reproduction

```bash
git clone https://github.com/ajinkyamaster/stellar-classification.git
cd stellar-classification
pip install -r requirements.txt
```

Run the full pipeline via the notebook in `notebooks/`, or the individual scripts in `src/` in order: feature engineering → base model training → stacking.

## Competition

[Kaggle Playground Series S6E6 — Predicting Stellar Class](https://www.kaggle.com/competitions/playground-series-s6e6)

## Author

**Ajinkya Nimbekar**
[GitHub](https://github.com/ajinkyamaster) · [LinkedIn](https://linkedin.com/in/ajinkya-nimbekar)
