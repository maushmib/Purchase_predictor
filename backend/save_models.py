import joblib
import pandas as pd
import os
import sys

# Ensure this script runs from the backend directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Append parent directory (Purchase_predictor) to path to load its modules
sys.path.append(os.path.abspath(".."))

from src.preprocessing import load_and_preprocess
from src.feature_engineering import select_marketing_signals
from src.modeling import train_decision_tree, train_bayes, train_regression

def run_and_save():
    os.makedirs("models_saved", exist_ok=True)
    
    print("[*] Preprocessing dataset...")
    # Change into Purchase_predictor to load relative datasets
    old_cwd = os.getcwd()
    os.chdir("..")
    X_train_sc, X_test_sc, y_train_cls, y_test_cls, y_train_reg, y_test_reg, feat_names, pipeline = load_and_preprocess()
    X_train_sel, X_test_sel, best_feats, selector = select_marketing_signals(X_train_sc, y_train_cls, X_test_sc, feat_names)
    os.chdir(old_cwd)
    
    print("[*] Training models...")
    dt_res = train_decision_tree(X_train_sel, y_train_cls, X_test_sel, y_test_cls)
    bayes_res = train_bayes(X_train_sel, y_train_cls, X_test_sel, y_test_cls)
    reg_res = train_regression(X_train_sel, y_train_reg, X_test_sel, y_test_reg)
    
    print("[*] Saving models to models_saved/ ...")
    joblib.dump(pipeline, "models_saved/pipeline.joblib")
    joblib.dump(selector, "models_saved/selector.joblib")
    joblib.dump(dt_res["model"], "models_saved/dt.joblib")
    joblib.dump(bayes_res["model"], "models_saved/bayes.joblib")
    joblib.dump(reg_res["model"], "models_saved/reg.joblib")
    
    print("[*] Success! Joblib binaries generated.")

if __name__ == "__main__":
    run_and_save()
