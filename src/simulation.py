import pandas as pd
import numpy as np

def simulate_scenario(base_row_df, pipeline, selector, tree_model, reg_model, changes_dict):
    """
    Scenario Simulation Engine (GAME-CHANGER)
    Takes a base Pandas DataFrame row, applies a dictionary of delta changes,
    pushes it through the preprocessing, feature selection, and outputs new predictions.
    
    Args:
        base_row_df: Single row of original features (DataFrame)
        pipeline: The fitted sklearn pipeline (ColumnTransformer)
        selector: Fitted SelectKBest
        tree_model: Trained Decision Tree (or any classifier)
        reg_model: Trained Regression model
        changes_dict: Dict of column: multiplier (e.g., {'PageValues': 1.2})
        
    Returns:
        new_prob: New probability of purchase (assumes binary class 1)
        new_spend: New predicted spending amount
    """
    # 1. Apply changes
    sim_row = base_row_df.copy()
    for col, multiplier in changes_dict.items():
        if col in sim_row.columns:
            # We multiply by the scaling multiplier (e.g., 20% increase -> 1.2)
            sim_row[col] = sim_row[col] * multiplier
    
    # 2. Preprocess using pipeline
    sim_scaled = pipeline.transform(sim_row)
    
    # 3. Apply selector for Tree (Assuming Tree uses selected features)
    sim_sel = selector.transform(sim_scaled)
    
    # 4. Predict probability
    if hasattr(tree_model, 'predict_proba'):
        new_prob = tree_model.predict_proba(sim_sel)[0][1] # Prob of class 1
    else:
        new_prob = float(tree_model.predict(sim_sel)[0])
        
    # 5. Predict revenue (Assuming Regression uses PCA, but for simplicity of simulation architecture 
    # we can pass it through a mocked PCA or assume linear model uses the selected features.)
    # In main.py we will align the regression to train on sim_sel or pass pca to this func.
    # To keep this generic, let's just assume reg_model uses sim_sel in our new upgraded pipeline.
    # We will ensure reg uses selected features in main.py for interpretability.
    new_spend = reg_model.predict(sim_sel)[0]
    
    return float(new_prob), float(max(0, new_spend))
