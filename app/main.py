import streamlit as st
import pandas as pd
from data_loader import load_data, validate_date
from model import load_model, predict_price, train_and_save_model, evaluate_model, train_test_split
from utils import plot_stock_price, plot_model_performance, validate_ticker, validate_numeric_input

# Set page config
st.set_page_config(
    page_title="Stock Market Prediction",
    page_icon=":chart_with_upwards_trend:",
    layout="wide"
)

# Title of the app
st.title("Stock Market Prediction App📊")
st.subheader("Using Random Forest🌳")

# Sidebar: Stock selection and date range
with st.sidebar:
    st.header("Stock Selection")
    stock_ticker = validate_ticker(st.text_input("Enter Stock Ticker Symbol", value='NVDA'))
    start_date = st.date_input("Start Date", value=pd.to_datetime("2022-01-01"))
    end_date = st.date_input("End Date", value=pd.to_datetime("2024-09-01"))

# Load and display data
with st.spinner("Fetching stock data..."):
    hist = load_data(stock_ticker, start_date, end_date)

if not hist.empty:
    st.success("Data successfully loaded!")
    st.write(f"Displaying data for: **{stock_ticker}**")

    # Display stock price chart
    fig = plot_stock_price(hist, stock_ticker)
    st.plotly_chart(fig)

    # Display historical data
    st.write("**Filtered Historical Data** (sorted by Date)")
    st.dataframe(hist.sort_values(by='Date'))

    # Model training and evaluation
    X = hist.drop(columns=['Date', 'Close', 'Adj Close'])
    y = hist['Close']

    regressor = load_model(stock_ticker)
    if regressor is None:
        regressor, X_test, y_test = train_and_save_model(X, y, stock_ticker)
    else:
        # If the model is loaded, we need to create X_test and y_test for evaluation
        _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Evaluate the model
    mse, rmse, mae, y_pred = evaluate_model(regressor, X_test, y_test)

    # Display model performance metrics
    st.subheader("Model Performance Metrics")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Mean Squared Error (MSE)", f"{mse:.4f}")
    with col2:
        st.metric("Root Mean Squared Error (RMSE)", f"{rmse:.4f}")
    with col3:
        st.metric("Mean Absolute Error (MAE)", f"{mae:.4f}")

    # Plot model performance
    plot_model_performance(y_test, y_pred)

    # Prediction inputs
    with st.sidebar:
        st.header("Prediction Inputs")
        open_price = validate_numeric_input(st.number_input("Open Price", min_value=0.0, step=0.1), "Open Price")
        high_price = validate_numeric_input(st.number_input("High Price", min_value=0.0, step=0.1), "High Price")
        low_price = validate_numeric_input(st.number_input("Low Price", min_value=0.0, step=0.1), "Low Price")
        volume = validate_numeric_input(st.number_input("Volume", min_value=0, step=1), "Volume")

    # Predict button
    if st.sidebar.button("Predict Closing Price"):
        if all([open_price, high_price, low_price, volume]):
            prediction = predict_price(regressor, open_price, high_price, low_price, volume)
            st.subheader(f"Predicted Closing Price for {stock_ticker}: {prediction:.2f}")

            # Model performance
            y_pred = regressor.predict(X)
            plot_model_performance(y, y_pred)
        else:
            st.error("Please enter valid values for all inputs.")
else:
    st.error(f"Unable to load data for {stock_ticker}. Please check the ticker symbol.")