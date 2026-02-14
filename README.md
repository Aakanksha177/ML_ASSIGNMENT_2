# Machine Learning Classification Models on Adult Income Dataset

## a. Problem Statement
The objective of this project is to build and compare multiple machine learning classification models to predict whether an individual earns more than $50K per year based on demographic and employment-related attributes. This is a binary classification problem where the target variable indicates income class (<=50K or >50K).

---

## b. Dataset Description
The Adult Income dataset is obtained from the UCI Machine Learning Repository. It contains demographic and employment-related attributes collected from census data, such as age, education, occupation, marital status, and working hours per week. The dataset is used to predict whether a person’s annual income exceeds $50K.

- Dataset Type: Binary Classification
- Number of Instances: ~48,000 (after preprocessing ~30,000+)
- Number of Features: 14 input features
- Target Variable: Income (<=50K or >50K)

---

## c. Models Used and Evaluation Metrics

The following machine learning classification models were implemented and evaluated using the same dataset. The evaluation metrics used for comparison are Accuracy, AUC Score, Precision, Recall, F1 Score, and Matthews Correlation Coefficient (MCC).

### Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---------------|----------|-----|-----------|--------|----|-----|
| Logistic Regression | 0.8175 | 0.8501 | 0.7135 | 0.4461 | 0.5490 | 0.4613 |
| Decision Tree | 0.8076 | 0.7437 | 0.6128 | 0.6165 | 0.6147 | 0.4864 |
| kNN | 0.8190 | 0.8498 | 0.6530 | 0.5826 | 0.6158 | 0.4993 |
| Naive Bayes | 0.7978 | 0.8498 | 0.6986 | 0.3302 | 0.4485 | 0.3798 |
| Random Forest (Ensemble) | 0.8545 | 0.9027 | 0.7480 | 0.6265 | 0.6819 | 0.5924 |
| XGBoost (Ensemble) | 0.8616 | 0.9204 | 0.7636 | 0.6431 | 0.6982 | 0.6131 |

---

### Observations on Model Performance

| ML Model Name | Observation about model performance |
|---------------|------------------------------------|
| Logistic Regression | Performs reasonably well but shows lower recall, indicating that linear decision boundaries are not sufficient to capture all high-income cases. |
| Decision Tree | Captures non-linear relationships and provides balanced precision and recall, but slightly lower accuracy due to possible overfitting. |
| kNN | Provides a good balance between precision and recall, but performance depends on distance calculations and scaling of features. |
| Naive Bayes | Fast and simple model, but lower recall suggests the independence assumption among features limits its performance. |
| Random Forest (Ensemble) | Improves overall performance with higher accuracy, AUC, and MCC by combining multiple decision trees and reducing overfitting. |
| XGBoost (Ensemble) | Achieves the best performance across most metrics, especially AUC and MCC, demonstrating strong capability to model complex feature interactions. |
