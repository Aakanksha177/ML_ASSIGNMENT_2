# Machine Learning Classification Models - Adult Income Dataset

## Problem Statement
The objective of this project is to predict whether an individual's income exceeds $50K per year based on demographic and employment-related attributes. This is a binary classification problem.

## Dataset Description
The Adult Income dataset is obtained from the UCI Machine Learning Repository. It contains demographic and employment information such as age, education, occupation, hours-per-week, etc. The target variable indicates whether income is <=50K or >50K.

- Number of Instances: ~48,000
- Number of Features: 14
- Type: Binary Classification

## Models Used and Evaluation Metrics

| ML Model | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|----------|----------|-----|-----------|--------|----------|-----|
| Logistic Regression | | | | | | |
| Decision Tree | | | | | | |
| KNN | | | | | | |
| Naive Bayes | | | | | | |
| Random Forest | | | | | | |
| XGBoost | | | | | | |

*(Fill values from your results table)*

## Observations

| ML Model | Observation |
|----------|-------------|
| Logistic Regression | Performs well on linear relationships but may underfit complex patterns. |
| Decision Tree | Captures nonlinear relationships but prone to overfitting. |
| KNN | Works well for local patterns but computationally expensive. |
| Naive Bayes | Fast and simple but assumes feature independence. |
| Random Forest | Provides strong performance by reducing overfitting using ensemble learning. |
| XGBoost | Generally achieves the best performance due to gradient boosting optimization. |

## Streamlit App Features
- Upload test CSV dataset
- Select model from dropdown
- Display evaluation metrics
- Show confusion matrix visualization

## Repository Structure
