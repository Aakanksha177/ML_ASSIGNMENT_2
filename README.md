# Machine Learning Classification Models on Adult Income Dataset

## a. Problem Statement
The objective of this project is to build and compare multiple machine learning classification models to predict whether an individual earns more than $50K per year based on demographic and employment-related attributes. This is a binary classification problem where the target variable indicates income class (<=50K or >50K).

---

## b. Dataset Description
The Adult Income dataset is obtained from the UCI Machine Learning Repository and is widely used for benchmarking classification algorithms. The dataset is derived from the 1994 U.S. Census database and contains demographic and socio-economic attributes of individuals. The goal is to predict whether a person’s annual income exceeds $50K based on these attributes.

### Dataset Characteristics
- Source: UCI Machine Learning Repository (Adult Census Income Dataset)
- Problem Type: Binary Classification
- Number of Instances: ~48,842 (after removing missing values ~30,000+ usable records)
- Number of Features: 14 input features + 1 target variable
- Target Variable: `income`
  - `<=50K` → Class 0
  - `>50K` → Class 1

### Feature Types
The dataset contains both numerical and categorical features:

- Numerical Features: age, fnlwgt, education_num, capital_gain, capital_loss, hours_per_week  
- Categorical Features: workclass, education, marital_status, occupation, relationship, race, sex, native_country  

Categorical variables were encoded using Label Encoding to convert them into numerical form suitable for machine learning algorithms.

### Data Preprocessing Steps
The following preprocessing steps were applied:
1. Handling missing values by removing rows containing unknown entries (“?”)
2. Encoding categorical features using LabelEncoder
3. Converting the target variable into binary numerical format (0 and 1)
4. Feature scaling applied for distance-based and linear models (Logistic Regression, kNN, Naive Bayes)

### Class Distribution
The dataset is moderately imbalanced, with a higher proportion of individuals earning <=50K compared to >50K. Therefore, evaluation metrics such as Precision, Recall, F1-score, and MCC are more informative than accuracy alone.

### Relevance of Dataset
This dataset is suitable for evaluating multiple classification algorithms because:
- It contains both linear and non-linear feature relationships
- It includes mixed data types (categorical + numerical)
- It is large enough to demonstrate performance differences between simple and ensemble models

Thus, it provides a realistic real-world classification scenario for comparing the performance of Logistic Regression, Decision Tree, kNN, Naive Bayes, Random Forest, and XGBoost models.

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
| kNN | Provides a good balance between precision and recall, but performance depends on distance calculations and feature scaling. |
| Naive Bayes | Fast and simple model, but lower recall suggests the independence assumption among features limits its performance. |
| Random Forest (Ensemble) | Improves overall performance with higher accuracy, AUC, and MCC by combining multiple decision trees and reducing overfitting. |
| XGBoost (Ensemble) | Achieves the best performance across most metrics, especially AUC and MCC, demonstrating strong capability to model complex feature interactions. |

<img width="1179" height="749" alt="Screenshot 2026-02-15 at 3 58 08 AM" src="https://github.com/user-attachments/assets/899ce00e-e050-41fc-9dab-2e3e68b4d93e" />

