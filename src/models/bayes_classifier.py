from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

def train_bayes(X_train, y_train, X_test, y_test):
    model = GaussianNB()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    return {
        "model": model,
        "accuracy": accuracy_score(y_test, y_pred),
        "predictions": y_pred
    }