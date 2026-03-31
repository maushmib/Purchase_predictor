import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.preprocess import load_and_preprocess
from src.feature_selection import select_features
from src.dimensionality import reduce_dimensionality
from src.models.bayes_classifier import train_bayes
from src.models.decision_tree import train_decision_tree
from src.models.regression import train_regression
from src.models.neural_network import train_nn
from src.evaluate import print_metrics

# --- Load & Preprocess ---
X_train, X_test, y_train, y_test = load_and_preprocess()

# --- Feature Selection ---
X_train_sel, X_test_sel = select_features(X_train, y_train, X_test)

# --- Dimensionality Reduction ---
dim_results = reduce_dimensionality(X_train_sel, X_test_sel)
X_train_pca, X_test_pca = dim_results["pca"]

# --- Function to plot predictions ---
def plot_predictions(y_true, y_pred, title):
    sns.scatterplot(x=range(len(y_true)), y=y_true, label="True", color="blue")
    sns.scatterplot(x=range(len(y_pred)), y=y_pred, label="Predicted", color="red")
    plt.title(title)
    plt.xlabel("Sample Index")
    plt.ylabel("Value / Class")
    plt.legend()
    plt.show()

# --- Train Bayesian Classifier ---
print("Training Bayesian Classifier...")
bayes_result = train_bayes(X_train_pca, y_train, X_test_pca, y_test)
bayes_preds = bayes_result["predictions"]
print("Bayes Accuracy:", bayes_result["accuracy"])
print_metrics(y_test, bayes_preds)
plot_predictions(y_test, bayes_preds, "Bayesian Classifier Predictions")

# --- Train Decision Tree ---
print("Training Decision Tree...")
dt_result = train_decision_tree(X_train_pca, y_train, X_test_pca, y_test)
dt_preds = dt_result["predictions"]
print("Decision Tree Accuracy:", dt_result["accuracy"])
print_metrics(y_test, dt_preds)
plot_predictions(y_test, dt_preds, "Decision Tree Predictions")

# --- Train Regression ---
print("Training Regression...")
reg_result = train_regression(X_train_pca, y_train, X_test_pca, y_test)
reg_preds = reg_result["predictions"]
print("Regression RMSE:", reg_result["rmse"])
plot_predictions(y_test, reg_preds, "Regression Predictions")

# --- Train Neural Network ---
print("Training Neural Network...")
nn_result = train_nn(X_train_pca, y_train, X_test_pca, y_test)
nn_preds = nn_result["predictions"]
print("Neural Network Accuracy:", nn_result["accuracy"])
print_metrics(y_test, nn_preds)
plot_predictions(y_test, nn_preds, "Neural Network Predictions")

# --- Save All Predictions ---
df_preds = pd.DataFrame({
    "True": y_test.values,
    "Bayes": bayes_preds,
    "DecisionTree": dt_preds,
    "NeuralNetwork": nn_preds,
    "Regression": reg_preds
})
df_preds.to_csv("predictions.csv", index=False)
print("Predictions saved to predictions.csv")

# --- PCA Scatter Plot ---
sns.scatterplot(x=X_test_pca[:, 0], y=X_test_pca[:, 1], hue=y_test)
plt.title("PCA Scatter Plot of Test Data")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()