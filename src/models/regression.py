from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np

def train_regression(X_train, y_train, X_test, y_test):
    """
    Trains a linear regression model to predict the target.
    
    Args:
        X_train: Training features
        y_train: Training target
        X_test: Testing features
        y_test: Testing target
    
    Returns:
        dict: {
            "model": trained LinearRegression model,
            "rmse": root mean squared error on test set
        }
    """
    # Initialize and train the regression model
    reg = LinearRegression()
    reg.fit(X_train, y_train)
    
    # Predict on test set
    y_pred = reg.predict(X_test)
    
    # Calculate RMSE (root mean squared error)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    return {
        "model": reg,
        "rmse": rmse,
        "predictions": y_pred
    }