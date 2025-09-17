"""
House Price Prediction - Training Script
----------------------------------------
This script trains a machine learning model to predict house prices
based on features like Area, Bedrooms, Bathrooms, Location, and YearBuilt.

Usage:
    python src/train.py
"""

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# -------------------------------
# Step 1: Load Dataset
# -------------------------------
def load_data():
    """
    Loads the dataset from data/processed/house_prices.csv
    (Replace with your dataset path if different).
    """
    dataset_path = "data/processed/house_prices.csv"

    if not os.path.exists(dataset_path):
        # Create a small sample dataset if file is missing
        print(f"[INFO] Dataset not found at {dataset_path}. Using sample dataset...")
        data = pd.DataFrame({
            'Area': [1200, 1800, 2500, 1500, 2200],
            'Bedrooms': [2, 3, 4, 2, 3],
            'Bathrooms': [2, 3, 3, 1, 2],
            'Location': ['Hyderabad', 'Bangalore', 'Chennai', 'Hyderabad', 'Bangalore'],
            'YearBuilt': [2015, 2010, 2018, 2012, 2016],
            'Price': [4500000, 7500000, 12000000, 5000000, 8000000]
        })
    else:
        print(f"[INFO] Loading dataset from {dataset_path}")
        data = pd.read_csv(dataset_path)

    return data


# -------------------------------
# Step 2: Preprocessing
# -------------------------------
def preprocess_data(data):
    encoder = LabelEncoder()
    data['Location'] = encoder.fit_transform(data['Location'])

    X = data[['Area', 'Bedrooms', 'Bathrooms', 'Location', 'YearBuilt']]
    y = data['Price']

    return X, y, encoder


# -------------------------------
# Step 3: Train Model
# -------------------------------
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # Evaluation
    print("✅ Model trained successfully!")
    print("R² Score:", r2_score(y_test, y_pred))
    print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))

    return model


# -------------------------------
# Step 4: Save Model
# -------------------------------
def save_model(model, encoder):
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/house_price_model.pkl")
    joblib.dump(encoder, "models/location_encoder.pkl")
    print("[INFO] Model and encoder saved in 'models/' directory.")


# -------------------------------
# Main
# -------------------------------
if __name__ == "__main__":
    data = load_data()
    X, y, encoder = preprocess_data(data)
    model = train_model(X, y)
    save_model(model, encoder)
