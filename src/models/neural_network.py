from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

def train_nn(X_train, y_train, X_test, y_test):
    model = Sequential([
        Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=30, batch_size=32, validation_split=0.2, verbose=1)

    # Evaluate
    loss, acc = model.evaluate(X_test, y_test, verbose=0)

    # Get predictions (convert probabilities to 0/1)
    y_pred = (model.predict(X_test) > 0.5).astype(int).flatten()

    return {
        "model": model,
        "accuracy": acc,
        "predictions": y_pred  # ✅ now y_pred is defined
    }