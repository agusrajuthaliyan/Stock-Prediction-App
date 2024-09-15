import yfinance as yf
import pandas as pd
import streamlit as st

@st.cache_data(ttl=86400)  # Cache data for 24 hours
def load_data(ticker, start_date, end_date):
    try:
        stock_data = yf.download(ticker, start=start_date, end=end_date)
        stock_data.reset_index(inplace=True)
        return stock_data
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
        return pd.DataFrame()

def validate_date(date_str):
    try:
        return pd.to_datetime(date_str)
    except ValueError:
        st.error("Invalid date format. Please use YYYY-MM-DD.")
        return None