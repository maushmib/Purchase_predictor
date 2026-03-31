import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from src.preprocess import load_and_preprocess
from src.feature_selection import select_features
from src.dimensionality import reduce_dimensionality
from src.models.bayes_classifier import train_bayes
from src.models.decision_tree import train_decision_tree
from src.models.regression import train_regression
from src.models.neural_network import train_nn

st.title("🛒 Smart E-Commerce Purchase Predictor Dashboard")

# Load and preprocess
X_train, X_test, y_train, y_test = load_and_preprocess()
X_train_sel, X_test_sel = select_features(X_train, y_train, X_test)
dim_results = reduce_dimensionality(X_train_sel, X_test_sel)
X_train_pca, X_test_pca = dim_results["pca"]

st.header("Model Training & Metrics")

# --- Bayesian Classifier ---
st.subheader("1️⃣ Bayesian Classifier")
bayes_result = train_bayes(X_train_pca, y_train, X_test_pca, y_test)
y_pred_bayes = bayes_result["model"].predict(X_test_pca)
st.write("Accuracy:", bayes_result["accuracy"])
st.text("Classification Report:\n" + classification_report(y_test, y_pred_bayes))

cm = confusion_matrix(y_test, y_pred_bayes)
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
ax.set_title("Confusion Matrix")
ax.set_xlabel("Predicted")
ax.set_ylabel("True")
st.pyplot(fig)

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred_bayes)
roc_auc = auc(fpr, tpr)
fig2, ax2 = plt.subplots()
ax2.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
ax2.plot([0,1], [0,1], "k--")
ax2.set_title("ROC Curve - Bayesian")
ax2.set_xlabel("False Positive Rate")
ax2.set_ylabel("True Positive Rate")
ax2.legend(loc="lower right")
st.pyplot(fig2)

# --- Decision Tree ---
st.subheader("2️⃣ Decision Tree")
dt_result = train_decision_tree(X_train_pca, y_train, X_test_pca, y_test)
y_pred_dt = dt_result["model"].predict(X_test_pca)
st.write("Accuracy:", dt_result["accuracy"])
st.text("Classification Report:\n" + classification_report(y_test, y_pred_dt))

cm = confusion_matrix(y_test, y_pred_dt)
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", ax=ax)
ax.set_title("Confusion Matrix")
ax.set_xlabel("Predicted")
ax.set_ylabel("True")
st.pyplot(fig)

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred_dt)
roc_auc = auc(fpr, tpr)
fig2, ax2 = plt.subplots()
ax2.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
ax2.plot([0,1], [0,1], "k--")
ax2.set_title("ROC Curve - Decision Tree")
ax2.set_xlabel("False Positive Rate")
ax2.set_ylabel("True Positive Rate")
ax2.legend(loc="lower right")
st.pyplot(fig2)

# --- Regression ---
st.subheader("3️⃣ Linear Regression")
reg_result = train_regression(X_train_pca, y_train, X_test_pca, y_test)
st.write("RMSE:", reg_result["rmse"])

# --- Neural Network ---
st.subheader("4️⃣ Neural Network")
nn_result = train_nn(X_train_pca, y_train, X_test_pca, y_test)
y_pred_nn = (nn_result["model"].predict(X_test_pca) > 0.5).astype(int).flatten()
st.write("Accuracy:", nn_result["accuracy"])
st.text("Classification Report:\n" + classification_report(y_test, y_pred_nn))

cm = confusion_matrix(y_test, y_pred_nn)
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Oranges", ax=ax)
ax.set_title("Confusion Matrix")
ax.set_xlabel("Predicted")
ax.set_ylabel("True")
st.pyplot(fig)

# --- PCA Scatter Plot ---
st.subheader("PCA Scatter Plot")
fig, ax = plt.subplots()
sns.scatterplot(x=X_test_pca[:,0], y=X_test_pca[:,1], hue=y_test)
ax.set_title("PCA of Test Data (True Labels)")
st.pyplot(fig)

# --- Save Predictions ---
st.subheader("Predictions Table")
pred_df = pd.DataFrame({
    "True": y_test,
    "Bayes": y_pred_bayes,
    "DecisionTree": y_pred_dt,
    "NeuralNetwork": y_pred_nn,
    "Regression": reg_result["model"].predict(X_test_pca)
})
st.dataframe(pred_df)
pred_df.to_csv("results/predictions.csv", index=False)
st.success("Predictions saved to results/predictions.csv")