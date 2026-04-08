# Lung Cancer Prediction

A machine learning pipeline that predicts lung cancer presence based on patient symptoms and lifestyle factors. Trained on clinical survey data with ADASYN oversampling to handle class imbalance, and evaluated across 4 classifiers with GridSearchCV tuning.

## How It Works

```
Patient Data → Preprocessing → Feature Engineering → ADASYN Oversampling → Model Training → Prediction
```

## Dataset

- **Size:** 309 records (276 after removing 33 duplicates)
- **Features:** 15 patient attributes (symptoms + lifestyle)
- **Target:** Lung cancer presence (Yes/No)
- **Class Distribution:** 86.2% positive, 13.8% negative — highly imbalanced
- **No missing values**

## Features

| Symptoms | Lifestyle |
|----------|-----------|
| Coughing | Smoking |
| Shortness of breath | Alcohol consuming |
| Wheezing | Peer pressure |
| Chest pain | |
| Swallowing difficulty | |
| Fatigue | |
| Yellow fingers | |
| Allergy | |
| Anxiety | |
| Chronic disease | |

## Key Preprocessing Steps

- **Label encoding** for binary features (2/1 → 1/0) and gender (M/F → 1/0)
- **Dropped** gender and age columns based on low correlation with target
- **Feature engineering:** Created `allergalcohol` interaction feature (allergy × alcohol consuming)
- **ADASYN oversampling** on minority class (276 → 472 samples) to address 86/14 class imbalance
- **Stratified train/test split** (80/20)

## Models & Results

| Model | Train Accuracy | Test Accuracy | AUROC |
|-------|---------------|---------------|-------|
| **Random Forest** | **97.88%** | **95.79%** | **0.9634** |
| K-Nearest Neighbors | 97.88% | 94.74% | 0.9639 |
| Logistic Regression | 93.90% | 94.74% | 0.9794 |
| Multinomial Naive Bayes | 72.15% | 66.32% | 0.7338 |

All models tuned with **10-fold cross-validation GridSearchCV** using ROC-AUC as scoring metric.

## Top Correlated Features with Lung Cancer

| Feature | Correlation |
|---------|-------------|
| Allergy | 0.334 |
| Alcohol consuming | 0.294 |
| Swallowing difficulty | 0.269 |
| Coughing | 0.253 |
| Wheezing | 0.249 |

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python |
| ML Models | Scikit-learn (RandomForest, KNN, LogisticRegression, MultinomialNB) |
| Oversampling | imbalanced-learn (ADASYN) |
| Visualization | Matplotlib, Seaborn |
| Data Processing | Pandas, NumPy |

## Project Structure

```
├── lung_cancer_prediction.ipynb    # Full pipeline notebook
├── lung_cancer_dataset.csv         # Dataset (309 records)
└── README.md
```

## Key Findings

- Random Forest achieved the best balance of accuracy (95.79%) and generalization
- Logistic Regression had the highest AUROC (0.9794), making it the best at ranking predictions
- Multinomial Naive Bayes performed poorly — the feature distributions don't fit the multinomial assumption
- Allergy and alcohol consumption are the strongest predictors, stronger than smoking
- ADASYN oversampling was critical — without it, models would be biased toward the majority class

## How to Run

```bash
pip install pandas numpy scikit-learn matplotlib seaborn imbalanced-learn feature-engine
jupyter notebook lung_cancer_prediction.ipynb
```

## License

Apache 2.0
