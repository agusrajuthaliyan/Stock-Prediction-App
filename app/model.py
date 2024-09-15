import os
import joblib
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pandas as pd
import numpy as np

def save_model(stock_ticker, model):
    if not os.path.exists('models'):
        os.makedirs('models')
    filename = f"models/{stock_ticker}_model.pkl"
    joblib.dump(model, filename)

def load_model(stock_ticker):
    filename = f"models/{stock_ticker}_model.pkl"
    if os.path.exists(filename):
        return joblib.load(filename)
    else:
        return None

def train_and_save_model(X, y, stock_ticker):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    with st.spinner("Training new model..."):
        regressor = RandomForestRegressor(n_estimators=100, random_state=42)
        regressor.fit(X_train, y_train)
    
    save_model(stock_ticker, regressor)
    st.success(f"Model training completed and saved for {stock_ticker}!")
    
    return regressor, X_test, y_test

def predict_price(model, open_price, high_price, low_price, volume):
    new_data = pd.DataFrame({
        'Open': [open_price],
        'High': [high_price],
        'Low': [low_price],
        'Volume': [volume]
    })
    prediction = model.predict(new_data)
    return prediction[0]

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    return mse, rmse, mae, y_pred