from sklearn.metrics import classification_report, confusion_matrix

def print_metrics(y_test, y_pred):
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))