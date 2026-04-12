import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def load_and_preprocess():
    """
    Loads raw data, derives a transparent proxy target for regression modelling,
    and returns scaled/encoded splits for our models.
    """
    df = pd.read_csv("data/online_shoppers_intention.csv")

    # -------------------------------------------------------------
    # 🎯 SYNTHETIC TARGET JUSTIFICATION:
    # The original dataset lacks an explicit "Total Revenue Amount" (only a boolean).
    # To facilitate the "How much will they spend?" Regression goal, we systematically 
    # synthesize a `SpendingAmount` metric out of their core engagement signals 
    # (`PageValues`). This serves as a proxy metric mirroring real-world proportional spend.
    # -------------------------------------------------------------
    np.random.seed(42)
    df["SpendingAmount"] = 0.0
    buy_mask = df["Revenue"] == True
    
    # Base purchase = $20. We add ~$5 per 1 point of PageValues, plus statistical noise.
    base_spending = 20.0 + (df.loc[buy_mask, "PageValues"] * 5.0) + np.random.normal(10, 5, size=buy_mask.sum())
    df.loc[buy_mask, "SpendingAmount"] = np.clip(base_spending, a_min=5.0, a_max=None)

    y_class = df["Revenue"].astype(int)
    y_reg = df["SpendingAmount"]
    
    X = df.drop(["Revenue", "SpendingAmount"], axis=1)

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

    X_train, X_test, y_train_class, y_test_class, y_train_reg, y_test_reg = train_test_split(
        X, y_class, y_reg, test_size=0.2, random_state=42
    )

    pipeline = Pipeline(steps=[("preprocess", preprocess)])
    X_train_scaled = pipeline.fit_transform(X_train)
    X_test_scaled = pipeline.transform(X_test)
    
    # Extract feature names properly after OneHotEncoding
    feat_names = num_cols.tolist() + pipeline.named_steps["preprocess"].transformers_[1][1].get_feature_names_out(cat_cols).tolist()

    return X_train_scaled, X_test_scaled, y_train_class, y_test_class, y_train_reg, y_test_reg, feat_names, pipeline
