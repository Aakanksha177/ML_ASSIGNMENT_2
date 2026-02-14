import pandas as pd
from sklearn.preprocessing import LabelEncoder

def load_and_preprocess_data(csv_path):
    columns = [
        "age", "workclass", "fnlwgt", "education", "education_num",
        "marital_status", "occupation", "relationship", "race", "sex",
        "capital_gain", "capital_loss", "hours_per_week", "native_country",
        "income"
    ]

    df = pd.read_csv(csv_path, header=None, names=columns, na_values=" ?")
    df.dropna(inplace=True)

    df["income"] = df["income"].apply(lambda x: 1 if x.strip() == ">50K" else 0)

    categorical_cols = df.select_dtypes(include=["object"]).columns
    label_encoders = {}

    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    X = df.drop("income", axis=1)
    y = df["income"]

    return X, y, label_encoders
