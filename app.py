import streamlit as st
import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix
)

st.set_page_config(page_title="Adult Income Classification", layout="wide")

st.title("Adult Income Classification Models")
st.write("Upload raw Adult dataset CSV and evaluate different ML models.")

# Column names (same as training)
columns = [
    "age", "workclass", "fnlwgt", "education", "education_num",
    "marital_status", "occupation", "relationship", "race", "sex",
    "capital_gain", "capital_loss", "hours_per_week", "native_country",
    "income"
]

# Model dropdown
model_choice = st.selectbox(
    "Select a Model",
    ["Logistic Regression", "Decision Tree", "KNN", "Naive Bayes", "Random Forest", "XGBoost"]
)

# Upload CSV
uploaded_file = st.file_uploader("Upload Test CSV", type=["csv"])

if uploaded_file is not None:
    # Load dataset
    df = pd.read_csv(uploaded_file, header=None, names=columns, na_values=" ?")

    st.subheader("Uploaded Dataset Preview")
    st.dataframe(df.head())

    # Drop missing values
    df.dropna(inplace=True)

    if df.shape[0] == 0:
        st.error("No valid rows after cleaning missing values.")
    else:
        # Convert target to binary
        df["income"] = df["income"].apply(lambda x: 1 if x.strip() == ">50K" else 0)

        # Encode categorical features
        categorical_cols = df.select_dtypes(include=["object"]).columns
        for col in categorical_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])

        # Split features and target
        X = df.drop("income", axis=1)
        y = df["income"]

        # Load selected model
        model_files = {
            "Logistic Regression": "saved_models/logistic.pkl",
            "Decision Tree": "saved_models/decision_tree.pkl",
            "KNN": "saved_models/knn.pkl",
            "Naive Bayes": "saved_models/naive_bayes.pkl",
            "Random Forest": "saved_models/random_forest.pkl",
            "XGBoost": "saved_models/xgboost.pkl"
        }

        model = joblib.load(model_files[model_choice])

        # Predict
        y_pred = model.predict(X)
        y_prob = model.predict_proba(X)[:, 1]

        # Metrics
        st.subheader("Evaluation Metrics")

        col1, col2, col3 = st.columns(3)
        col1.metric("Accuracy", f"{accuracy_score(y, y_pred):.4f}")
        col2.metric("AUC Score", f"{roc_auc_score(y, y_prob):.4f}")
        col3.metric("Precision", f"{precision_score(y, y_pred):.4f}")

        col4, col5, col6 = st.columns(3)
        col4.metric("Recall", f"{recall_score(y, y_pred):.4f}")
        col5.metric("F1 Score", f"{f1_score(y, y_pred):.4f}")
        col6.metric("MCC", f"{matthews_corrcoef(y, y_pred):.4f}")

        # Confusion Matrix
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y, y_pred)

        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)
