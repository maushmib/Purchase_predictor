import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def load_and_preprocess():
    df = pd.read_csv("data/online_shoppers_intention.csv")

    X = df.drop("Revenue", axis=1)
    y = df["Revenue"].astype(int)

    cat_cols = X.select_dtypes(include=["object", "bool"]).columns
    num_cols = X.select_dtypes(include=["int64", "float64"]).columns

    categorical_transformer = OneHotEncoder(handle_unknown="ignore")
    numerical_transformer = StandardScaler()

    preprocess = ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, num_cols),
            ("cat", categorical_transformer, cat_cols),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    pipeline = Pipeline(steps=[("preprocess", preprocess)])
    X_train = pipeline.fit_transform(X_train)
    X_test = pipeline.transform(X_test)

    return X_train, X_test, y_train, y_test