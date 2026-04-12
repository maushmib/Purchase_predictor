from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, mean_squared_error
import numpy as np

def train_decision_tree(X_train, y_train, X_test, y_test):
    tree = DecisionTreeClassifier(max_depth=6, random_state=42)
    tree.fit(X_train, y_train)
    y_pred = tree.predict(X_test)
    y_prob = tree.predict_proba(X_test) if hasattr(tree, "predict_proba") else None
    
    return {
        "model": tree,
        "accuracy": accuracy_score(y_test, y_pred),
        "predictions": y_pred,
        "probabilities": y_prob
    }

def train_bayes(X_train, y_train, X_test, y_test):
    model = GaussianNB()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)

    return {
        "model": model,
        "accuracy": accuracy_score(y_test, y_pred),
        "predictions": y_pred,
        "probabilities": y_prob
    }

def train_regression(X_train, y_train, X_test, y_test):
    reg = LinearRegression()
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    return {
        "model": reg,
        "rmse": rmse,
        "predictions": y_pred
    }
