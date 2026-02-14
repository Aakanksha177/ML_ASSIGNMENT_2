import streamlit as st
import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef

st.title("ML Classification Models - Adult Income Dataset")

# Model selection dropdown
model_choice = st.selectbox(
    "Select a Model",
    ["Logistic Regression", "Decision Tree", "KNN", "Naive Bayes", "Random Forest", "XGBoost"]
)

# Upload test CSV
uploaded_file = st.file_uploader("Upload Test CSV", type=["csv"])

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.write("Uploaded Data Preview:")
    st.dataframe(data.head())

    # Load corresponding model
    model_paths = {
        "Logistic Regression": "saved_models/logistic.pkl",
        "Decision Tree": "saved_models/decision_tree.pkl",
        "KNN": "saved_models/knn.pkl",
        "Naive Bayes": "saved_models/naive_bayes.pkl",
        "Random Forest": "saved_models/random_forest.pkl",
        "XGBoost": "saved_models/xgboost.pkl"
    }

    model = joblib.load(model_paths[model_choice])

    # Separate features and target (assuming last column is target)
    X_test = data.iloc[:, :-1]
    y_test = data.iloc[:, -1]

    # Predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Metrics
    st.subheader("Evaluation Metrics")
    st.write("Accuracy:", accuracy_score(y_test, y_pred))
    st.write("AUC Score:", roc_auc_score(y_test, y_prob))
    st.write("Precision:", precision_score(y_test, y_pred))
    st.write("Recall:", recall_score(y_test, y_pred))
    st.write("F1 Score:", f1_score(y_test, y_pred))
    st.write("MCC:", matthews_corrcoef(y_test, y_pred))

    # Confusion Matrix
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)
