import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, roc_curve, auc, precision_recall_curve, confusion_matrix

from src.preprocessing import load_and_preprocess
from src.feature_engineering import select_marketing_signals, build_personas, interpret_personas
from src.modeling import train_decision_tree, train_bayes, train_regression
from src.explainability import get_tree_shap_explainer, get_linear_shap_explainer, generate_global_importance_plot, generate_local_explanation
from src.simulation import simulate_scenario
from main import generate_business_insights

st.set_page_config(page_title="Decision Intelligence Platform", layout="wide")
st.title("🧠 E-Commerce Decision Intelligence Platform")
st.markdown("Predict behavior, explain predictions, simulate outcomes, and generate actionable insights.")

# --- System Booting ---
@st.cache_data
def load_all_engines():
    X_train_sc, X_test_sc, y_train_cls, y_test_cls, y_train_reg, y_test_reg, feat_names, pipeline = load_and_preprocess()
    X_train_sel, X_test_sel, best_feats, selector = select_marketing_signals(X_train_sc, y_train_cls, X_test_sc, feat_names)
    dim_res = build_personas(X_train_sel, X_test_sel)
    
    dt_res = train_decision_tree(X_train_sel, y_train_cls, X_test_sel, y_test_cls)
    bayes_res = train_bayes(X_train_sel, y_train_cls, X_test_sel, y_test_cls)
    reg_res = train_regression(X_train_sel, y_train_reg, X_test_sel, y_test_reg)
    
    # Generate static explainers
    tree_exp, tree_shap = get_tree_shap_explainer(dt_res["model"], X_train_sel)
    lin_exp, lin_shap = get_linear_shap_explainer(reg_res["model"], X_train_sel)
    
    # Reload raw for insights
    raw_df = pd.read_csv("data/online_shoppers_intention.csv")
    insights = generate_business_insights(raw_df, raw_df["Revenue"].astype(int))
    
    return {
        "X_train_sel": X_train_sel, "X_test_sel": X_test_sel, "best_feats": best_feats,
        "y_test_cls": y_test_cls, "y_test_reg": y_test_reg,
        "dt_res": dt_res, "bayes_res": bayes_res, "reg_res": reg_res,
        "dim_res": dim_res, "pipeline": pipeline, "selector": selector,
        "tree_exp": tree_exp, "tree_shap": tree_shap, "lin_exp": lin_exp, "lin_shap": lin_shap,
        "insights": insights, "raw_df": raw_df
    }

data = load_all_engines()

# Insights banner
st.info(f"💡 **Automated Business Insight:** {data['insights'][0]}")

tab1, tab2, tab3, tab4 = st.tabs([
    "1️⃣ Purchase Classifier & Explainability", 
    "2️⃣ Revenue Predictor", 
    "3️⃣ Customer Personas",
    "4️⃣ Scenario Simulator"
])

with tab1:
    st.header("Will user buy? & Why?")
    col_metrics, col_eval = st.columns(2)
    
    with col_metrics:
        st.subheader("Model Performance")
        st.write(f"**Decision Tree Accuracy:** {data['dt_res']['accuracy']:.2%}")
        st.write(f"**Bayes Probabilistic Accuracy:** {data['bayes_res']['accuracy']:.2%}")
        
        st.markdown("### Confusion Matrix (Decision Tree)")
        cm = confusion_matrix(data['y_test_cls'], data['dt_res']['predictions'])
        fig_cm, ax_cm = plt.subplots(figsize=(4, 3))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', ax=ax_cm)
        st.pyplot(fig_cm)
        
    with col_eval:
        st.subheader("Evaluation Enhancements")
        fig_eval, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(10, 4))
        
        # ROC
        fpr, tpr, _ = roc_curve(data['y_test_cls'], data['bayes_res']['probabilities'][:, 1])
        ax_roc.plot(fpr, tpr, label=f"Bayes AUC = {auc(fpr, tpr):.2f}")
        ax_roc.plot([0,1], [0,1], 'k--')
        ax_roc.set_title("ROC Curve")
        ax_roc.legend()
        
        # PR Curve
        prec, rec, _ = precision_recall_curve(data['y_test_cls'], data['dt_res']['probabilities'][:, 1] if data['dt_res']['probabilities'] is not None else data['dt_res']['predictions'])
        ax_pr.plot(rec, prec, color='green')
        ax_pr.set_title("Precision-Recall (DT)")
        ax_pr.set_xlabel('Recall')
        ax_pr.set_ylabel('Precision')
        st.pyplot(fig_eval)

    st.divider()
    st.subheader("Global Explainability (Decision Tree)")
    st.markdown("SHAP Summary reveals the most impactful factors pushing users to buy.")
    generate_global_importance_plot(data['tree_exp'], data['tree_shap'], data['X_train_sel'], data['best_feats'])

with tab2:
    st.header("Revenue Predictor & Explainability")
    st.write(f"**Regression RMSE:** ${data['reg_res']['rmse']:.2f}")
    
    st.markdown("### Global Feature Importance for Revenue Amount")
    st.markdown("*Note: SpendingAmount is a proxy metric systematically synthesized from PageValues and interaction depth. We generate SHAP on this regression output.*")
    generate_global_importance_plot(data['lin_exp'], data['lin_shap'], data['X_train_sel'], data['best_feats'])

with tab3:
    st.header("Customer Persona Interpretation")
    st.markdown("Factor Analysis isolates behavioral intent into 3 distinct components, automatically labeled by our interpretation engine.")
    
    fa_data = data['dim_res']['fa'][1]
    labels, expl = interpret_personas(fa_data)
    
    # Show scatter
    fig_pers, ax_p = plt.subplots(figsize=(7, 4))
    scatter = ax_p.scatter(fa_data[:, 0], fa_data[:, 1], c=data['y_test_cls'], cmap='coolwarm', alpha=0.5)
    ax_p.set_xlabel("Factor 1 (Intent Volume)")
    ax_p.set_ylabel("Factor 2 (Price Sensitivity / Exit Velocity)")
    ax_p.set_title("Persona Landscape")
    st.pyplot(fig_pers)
    
    st.subheader("Sample Mapped Personas")
    sample_df = pd.DataFrame({
        "Generated Persona": labels[:5],
        "System Explanation": expl[:5]
    })
    st.dataframe(sample_df)

with tab4:
    st.header("Scenario Simulator")
    st.markdown("If a user interacts differently, how does probability and spend change?")
    
    base_row = data['raw_df'].iloc[5]
    cols_to_simulate = ["PageValues", "BounceRates", "Informational_Duration"]
    
    col_sl1, col_sl2, col_sl3 = st.columns(3)
    p_val_mul = col_sl1.slider("PageValues Multiplier", 0.1, 3.0, 1.0)
    b_rate_mul = col_sl2.slider("BounceRates Multiplier", 0.1, 3.0, 1.0)
    i_dur_mul = col_sl3.slider("Info Duration Multiplier", 0.1, 3.0, 1.0)
    
    # Original stats calculated natively
    orig_prob, orig_spend = simulate_scenario(base_row.to_frame().T, data['pipeline'], data['selector'], data['dt_res']['model'], data['reg_res']['model'], {})
    
    if st.button("Simulate Impact"):
        changes = {"PageValues": p_val_mul, "BounceRates": b_rate_mul, "Informational_Duration": i_dur_mul}
        new_prob, new_spend = simulate_scenario(base_row.to_frame().T, data['pipeline'], data['selector'], data['dt_res']['model'], data['reg_res']['model'], changes)
        
        st.subheader("Simulation Results")
        m1, m2 = st.columns(2)
        m1.metric("Purchase Probability", f"{new_prob:.1%}", delta=f"{(new_prob - orig_prob):.1%}")
        m2.metric("Predicted Spend", f"${new_spend:.2f}", delta=f"${(new_spend - orig_spend):.2f}")
    else:
        st.write("Adjust multipliers and click simulate.")
