from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

def train_decision_tree(X_train, y_train, X_test, y_test):
    tree = DecisionTreeClassifier(max_depth=6)
    tree.fit(X_train, y_train)
    y_pred = tree.predict(X_test)

    return {
        "model": tree,
        "accuracy": accuracy_score(y_test, y_pred),
        "predictions": y_pred
    }