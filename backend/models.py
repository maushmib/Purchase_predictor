import joblib
import pandas as pd
import os

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models_saved")

class ModelRegistry:
    def __init__(self):
        self.pipeline = None
        self.selector = None
        self.dt = None
        self.bayes = None
        self.reg = None
        self.loaded = False

    def load_engines(self):
        if self.loaded:
            return
            
        print("[!] Loading pickled models from disk (without retraining)...")
        self.pipeline = joblib.load(os.path.join(MODELS_DIR, "pipeline.joblib"))
        self.selector = joblib.load(os.path.join(MODELS_DIR, "selector.joblib"))
        self.dt = joblib.load(os.path.join(MODELS_DIR, "dt.joblib"))
        self.bayes = joblib.load(os.path.join(MODELS_DIR, "bayes.joblib"))
        self.reg = joblib.load(os.path.join(MODELS_DIR, "reg.joblib"))
        self.loaded = True
        print("[!] Models loaded successfully.")

    def predict(self, features_dict: dict):
        if not self.loaded:
            self.load_engines()

        row_df = pd.DataFrame([features_dict])
        
        # Preprocess
        scaled = self.pipeline.transform(row_df)
        selected = self.selector.transform(scaled)
        
        # Predict
        dt_prob = float(self.dt.predict_proba(selected)[0][1])
        bayes_prob = float(self.bayes.predict_proba(selected)[0][1])
        spend = float(max(0.0, self.reg.predict(selected)[0]))
        
        # Tier classification logic
        if dt_prob > 0.78 and spend > 120:
            tier = "High Intent" # Originally VIP
        elif dt_prob > 0.55:
            tier = "High Intent"
        elif dt_prob > 0.30:
            tier = "At Risk"
        else:
            tier = "Window Shopper"
            
        return {
            "purchase_probability": dt_prob,
            "bayes_probability": bayes_prob,
            "predicted_revenue": spend,
            "final_prediction": tier
        }

registry = ModelRegistry()
