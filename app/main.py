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

    # # Understanding MSE and Feedback
    # st.subheader("Understanding the Evaluation Metric (MSE)")
    # st.write("""
    # Mean Squared Error (MSE) is a metric that tells us how far off our model's predictions are from the actual values.
    # A lower MSE value indicates better accuracy, as it means the predicted stock prices are closer to the real prices.

    # - **MSE measures the average squared difference between predicted values and actual values.**
    # - The closer MSE is to zero, the better the model's performance.

    # While a low MSE suggests that the model is performing well, it's important to keep in mind that no model is perfect,
    # especially in highly volatile areas like stock market prediction. This model should be used as a guide, not as a foolproof prediction tool.
    # """)

    # # Dynamic feedback based on MSE value
    # if mse < 10:
    #     st.success("The model's performance is quite good! The MSE is low, meaning the model predictions are fairly close to actual values. You can rely on this model for general guidance.")
    # elif mse < 50:
    #     st.warning("The model is somewhat accurate but shows room for improvement. It’s best to treat the predictions cautiously, especially during market fluctuations.")
    # else:
    #     st.error("The model’s performance is not optimal. The high MSE indicates large errors in predictions. It might not be safe to rely heavily on this model for decision-making.")

        # Understanding MSE and Feedback
    with st.expander("Understanding the Evaluation Metric (MSE)", expanded=False):
        st.markdown("""
        **Mean Squared Error (MSE)** is a key metric that helps evaluate the performance of the prediction model.

        - MSE measures the **average squared difference** between the actual and predicted values.
        - A **lower MSE** indicates better model accuracy, with predictions closer to the real stock prices.
        
        ### Quick Guide:
        - **MSE ≈ 0**: Excellent model performance, predictions are highly accurate.
        - **MSE > 50**: The model struggles with accurate predictions, especially in volatile markets.
        
        > _Note: Even a low MSE cannot fully guarantee perfect predictions in stock markets due to inherent volatility._
        """)

    # Dynamic feedback based on MSE value
    st.subheader("Model Reliability Feedback")
    col1, col2, col3 = st.columns([1, 4, 1])

    with col2:
        if mse < 10:
            st.success("✅ The model is performing **very well** with a low MSE. Predictions are quite reliable for general guidance.")
        elif mse < 50:
            st.warning("⚠️ The model is **moderately accurate**, but there is room for improvement. Be cautious, especially in volatile market conditions.")
        else:
            st.error("❌ The model's MSE is **high**, indicating large errors in predictions. It's not safe to rely heavily on this model.")


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

# Add a disclaimer note
st.markdown("""
    ---
    **Important Note:**  
    This app utilizes a **Random Forest** model for predicting stock prices, which is just one approach to financial forecasting.  
    - **Do not** rely solely on this tool for making financial decisions without understanding the market deeply.  
    - Stock markets are unpredictable, and no model can guarantee future prices accurately.
    
    _I, Agus Raju Thaliyan, am not responsible for any financial losses incurred by using this app. This is simply a predictive tool meant for learning and exploratory purposes._
""")

# footer section
st.markdown("""
    <style>
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: calc(100% - 250px);  /* Adjust width based on sidebar width (250px) */
        background-color: #ffffff;
        padding: 10px 0;
        font-family: 'Arial', sans-serif;
        font-size: 14px;
        color: #4d4d4d;
        border-top: 1px solid #eaeaea;
        display: flex;
        justify-content: center;  /* Centers the content horizontally */
        align-items: center;      /* Centers the content vertically */
        box-sizing: border-box;   /* Ensures padding fits well within the width */
        margin-left: 250px;       /* Offset the left margin to account for sidebar width */
    }
    .footer-content {
        text-align: center;
    }
    .footer a {
        color: #0073b1;  /* LinkedIn color */
        text-decoration: none;
        font-weight: bold;
        margin: 0 10px;
    }
    .footer a:hover {
        text-decoration: underline;
    }
    .footer p {
        margin: 0;
    }
    .footer-line {
        display: block;         /* Forces a new line */
    }
    @media (max-width: 768px) {
        .footer {
            flex-direction: column;  /* Stack content vertically on smaller screens */
            width: 100%;             /* Full width on smaller screens */
            margin-left: 0;         /* Remove left margin on smaller screens */
        }
        .footer a {
            display: block;          /* Stack links on mobile */
            margin: 5px 0;           /* Add margin between links */
        }
    }
    </style>
    <div class="footer">
        <div class="footer-content">
            <p>Made with ❤️ by <b>Agus Raju Thaliyan</b><span class="footer-line"></span> 
            <a href="https://www.linkedin.com/in/agusrajuthaliyan" target="_blank">LinkedIn</a> | 
            <a href="mailto:agusraju43@gmail.com">agusraju43@gmail.com</a></p>
        </div>
    </div>
""", unsafe_allow_html=True)

