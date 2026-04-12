import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

from src.preprocessing import load_and_preprocess
from src.feature_engineering import select_marketing_signals, build_personas, interpret_personas
from src.modeling import train_decision_tree, train_bayes, train_regression

def generate_business_insights(X_df, y_class):
    """
    Business Insight Generator (Item 4)
    Computes simple insights dynamically based on target splits.
    """
    insights = []
    
    # 1. PageValues Insight
    if "PageValues" in X_df.columns:
        high_pv = X_df[X_df["PageValues"] > X_df["PageValues"].median()]
        high_pv_conv = y_class.loc[high_pv.index].mean()
        all_conv = y_class.mean()
        multiplier = high_pv_conv / (all_conv + 1e-5)
        insights.append(f"Users with high PageValues have {multiplier:.1f}x higher conversion rate.")

    # 2. BounceRates Insight
    if "BounceRates" in X_df.columns:
        low_bounce = X_df[X_df["BounceRates"] < X_df["BounceRates"].median()]
        low_bounce_conv = y_class.loc[low_bounce.index].mean()
        insights.append(f"Users with below-average BounceRates convert at {low_bounce_conv:.1%}.")
        
    return insights

if __name__ == "__main__":
    print("==================================================")
    print("E-Commerce Decision Intelligence Platform - Engine")
    print("==================================================")

    # --- 1. Load & Preprocess ---
    print("\n[Phase 1] Preprocessing Data & Generating Synthetic Revenue Targets...")
    X_train_sc, X_test_sc, y_train_cls, y_test_cls, y_train_reg, y_test_reg, feat_names, pipeline = load_and_preprocess()

    # --- 2. Feature Engineering ---
    print("[Phase 2] Engineering Features & Constructing Personas...")
    X_train_sel, X_test_sel, best_feats, selector = select_marketing_signals(
        X_train_sc, y_train_cls, X_test_sc, feat_names
    )
    dim_res = build_personas(X_train_sel, X_test_sel)
    
    # Generate static insights (Using X_train combined for statical validity)
    try:
        raw_df = pd.read_csv("data/online_shoppers_intention.csv")
        y_overall = raw_df["Revenue"].astype(int)
        insights = generate_business_insights(raw_df, y_overall)
        print("\n=== Business Insights Generator ===")
        for ins in insights:
            print(f" 💡 {ins}")
    except Exception as e:
        print("Could not generate text insights.")

    # --- 3. Modeling ---
    print("\n[Phase 3] Booting Modeling Engines...")
    
    # Classification
    dt_res = train_decision_tree(X_train_sel, y_train_cls, X_test_sel, y_test_cls)
    bayes_res = train_bayes(X_train_sel, y_train_cls, X_test_sel, y_test_cls)
    
    print(f"--> Decision Tree Accuracy: {dt_res['accuracy']:.2%}")
    print(f"--> Bayesian Probability Modeler Accuracy: {bayes_res['accuracy']:.2%}")

    # Regression
    reg_res = train_regression(X_train_sel, y_train_reg, X_test_sel, y_test_reg)
    print(f"--> Revenue Predictor RMSE: ${reg_res['rmse']:.2f}")

    print("\n[Engine Ready] Transition to Streamlit (app.py) for Interactive Explainability, Simulation, and Persona maps.")
