import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
import joblib

def train():
    print("Downloading dataset...")
    url = "https://raw.githubusercontent.com/sumit0072/Car-Price-Prediction-Project/master/car%20data.csv"
    df = pd.read_csv(url)
    
    print("Dataset loaded. Shape:", df.shape)
    print("Columns:", df.columns.tolist())
    
    # Preprocessing
    # 1. Age of car
    # The dataset contains purchase year (e.g., 2014). We calculate Age relative to 2020 (the year this dataset was created/compiled)
    # so that the resulting 'Age' column aligns with typical actual age values in years.
    df['Age'] = 2020 - df['Year']
    
    # 2. Map categorical columns to numerical values matching the screenshot inputs:
    # Fuel Type (0: Petrol, 1: Diesel, 2: CNG)
    fuel_mapping = {'Petrol': 0, 'Diesel': 1, 'CNG': 2}
    df['Fuel_Type'] = df['Fuel_Type'].map(fuel_mapping)
    
    # Seller Type (0: Dealer, 1: Individual)
    seller_mapping = {'Dealer': 0, 'Individual': 1}
    df['Seller_Type'] = df['Seller_Type'].map(seller_mapping)
    
    # Transmission (0: Manual, 1: Auto/Automatic)
    transmission_mapping = {'Manual': 0, 'Automatic': 1}
    df['Transmission'] = df['Transmission'].map(transmission_mapping)
    
    # Select features and target
    # Target: Selling_Price (which represents the price in Lakhs)
    # Features: Present_Price, Kms_Driven, Fuel_Type, Seller_Type, Transmission, Owner, Age
    features = ['Present_Price', 'Kms_Driven', 'Fuel_Type', 'Seller_Type', 'Transmission', 'Owner', 'Age']
    X = df[features]
    y = df['Selling_Price']
    
    print("\nFeatures distribution:")
    print(X.describe())
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train Random Forest Regressor
    print("\nTraining Random Forest Regressor...")
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate model
    train_predictions = model.predict(X_train)
    test_predictions = model.predict(X_test)
    
    train_r2 = metrics.r2_score(y_train, train_predictions)
    test_r2 = metrics.r2_score(y_test, test_predictions)
    
    train_mae = metrics.mean_absolute_error(y_train, train_predictions)
    test_mae = metrics.mean_absolute_error(y_test, test_predictions)
    
    print(f"\nTraining Performance:")
    print(f"R2 Score: {train_r2:.4f}")
    print(f"MAE: {train_mae:.4f} Lakhs")
    
    print(f"\nTesting Performance:")
    print(f"R2 Score: {test_r2:.4f}")
    print(f"MAE: {test_mae:.4f} Lakhs")
    
    # Save model and feature metadata
    model_data = {
        'model': model,
        'features': features,
        'fuel_mapping': fuel_mapping,
        'seller_mapping': seller_mapping,
        'transmission_mapping': transmission_mapping
    }
    
    joblib.dump(model_data, "car_price_model.pkl")
    print("\nModel saved successfully to 'car_price_model.pkl'")

if __name__ == "__main__":
    train()
