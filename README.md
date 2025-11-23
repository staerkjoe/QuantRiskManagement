# Credit Risk Modeling for CopenCred

This project focuses on credit risk management using Machine Learning
models to refine lending decisions for a hypothetical financial
institution, CopenCred. The goal is to reduce credit risk, improve
borrower selection, align interest rates with risk, and establish
effective monitoring protocols.

Two model types are compared:

-   Logistic Regression (transparent baseline)
-   XGBoost (captures non-linear effects)

Dataset used: UCI Statlog (German Credit).

------------------------------------------------------------------------

# Data

| Property       | Value                         |
|----------------|-------------------------------|
| Dataset Size   | 1000 rows                     |
| Features       | 20 categorical + numeric      |
| Missing Data   | None                          |
| Target         | Binary (creditworthiness)     |
| Target Mapping | 0 = Bad, 1 = Good             |


------------------------------------------------------------------------

# Running the Training Scripts

## Preprocessing Summary

### Logistic Regression

-   One-hot encoding for nominal variables
-   Ordinal encoding for ordered categories
-   Standard scaling
-   SMOTE applied to training only

### XGBoost

-   Ordinal encoding
-   No scaling
-   No SMOTE (handles imbalance internally)

## Validation and Tuning

-   Stratified k-fold cross-validation
-   Randomized search followed by grid search
-   Objective: maximize precision for the "Good" class subject to recall
    constraints

## Commands

``` bash
python train_logistic_regression.py
python train_xgboost.py
```

------------------------------------------------------------------------

# Configuration and Parameters

| Parameter Category | Parameters | Why Change It |
|-------------------|------------|----------------|
| Model Validation | `k_folds`, `random_seed` | Ensures robust and repeatable cross-validation. |
| Hyperparameters | E.g., `log_reg__C`, `xgb__n_estimators`, `xgb__max_depth` | Controls model complexity and regularization. |
| Optimization Goal | `optimization_metric` (precision, f1_score, auc) | Aligns objective with business priorities. |
| Preprocessing | `smote_sampling_strategy`, `feature_engineering_ratios` | Handles class imbalance and feature shaping. |
| Decision Threshold | `final_threshold` | Adjusts precision–recall tradeoff. |


------------------------------------------------------------------------

# Key Findings and Business Impact

## Model Performance

| Model                         | Optimization | Precision | F1    | AUC   | Accuracy | Recall |
|-------------------------------|--------------|-----------|-------|-------|----------|--------|
| Logistic Regression (SMOTE)   | Precision/F1 | 0.754     | 0.838 | 0.782 | 0.745    | 0.943  |
| XGBoost                       | Precision    | 0.788     | 0.831 | 0.786 | 0.750    | 0.879  |


### Summary

-   Both models performed strongly with similar AUC.
-   SMOTE was essential for Logistic Regression.
-   XGBoost delivered the strongest precision and balanced recall.

------------------------------------------------------------------------

# Suitability and Risks

## Transparency vs. Complexity

-   Logistic Regression is easier to explain and compliant-friendly.
-   XGBoost captures non-linear patterns but is less transparent.

## Bias Risk

-   "Personal status/sex" appeared highly influential in XGBoost.
-   Requires fairness analysis and possibly feature restriction.

## Data Limitations

-   Dataset is small (1000 rows), limiting generalizability.

------------------------------------------------------------------------

# Governance and Controls

Required under the EU AI Act (credit scoring = high-risk AI):

-   Independent Validation
-   Ongoing Monitoring (accuracy, drift)
-   Periodic Retraining
-   Explainability (SHAP, LIME)
-   Strong Documentation and Version Control
