import shap
import matplotlib.pyplot as plt
import streamlit as st

def get_tree_shap_explainer(model, X_train):
    """
    Returns a SHAP TreeExplainer and shap_values for a tree-based model.
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train)
    return explainer, shap_values

def get_linear_shap_explainer(model, X_train):
    """
    Returns a SHAP LinearExplainer for regression.
    """
    explainer = shap.LinearExplainer(model, X_train)
    shap_values = explainer.shap_values(X_train)
    return explainer, shap_values

def generate_global_importance_plot(explainer, shap_values, X, feature_names=None):
    """
    Generates a SHAP summary plot for global importance.
    """
    fig = plt.figure()
    shap.summary_plot(shap_values, X, feature_names=feature_names, show=False)
    st.pyplot(fig)

def generate_local_explanation(explainer, shap_value_single, feature_names=None):
    """
    Generates a SHAP waterfall or force plot for local explanation.
    """
    # Create the waterfall plot figure
    fig, ax = plt.subplots()
    try:
        # shap 0.40+ waterfall formatting
        shap.plots.waterfall(shap_value_single, show=False)
    except Exception as e:
        st.write("Could not generate waterfall. Ensure SHAP object format is correct.")
    st.pyplot(fig)
