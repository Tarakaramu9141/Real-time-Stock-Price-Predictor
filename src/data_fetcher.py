import os
import pandas as pd
import yfinance as yf
from alpha_vantage.timeseries import TimeSeries

class DataFetcher:
    def __init__(self, api_key='YOUR_API_KEY'):
        self.api_key = api_key
        try:
            self.ts = TimeSeries(key=self.api_key, output_format='pandas')
        except:
            self.ts = None
        
    def fetch_historical_data(self, symbol, save_path='data/stock_data.csv'):
        """Fetch historical stock data using Yahoo Finance"""
        try:
            # Ensure data directory exists
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            # Use Yahoo Finance
            data = yf.download(symbol, start='2010-01-01', progress=False)
            
            if not data.empty:
                # Use Close price instead of Adj Close
                data = data[['Close']].copy()
                data.columns = ['Price']
                data.index.name = 'Date'
                data.to_csv(save_path)
                return data
            
            raise ValueError("Yahoo Finance returned empty data")
            
        except Exception as e:
            print(f"Error fetching data: {e}")
            return None

    def fetch_real_time_data(self, symbol):
        """Fetch real-time stock data"""
        try:
            stock = yf.Ticker(symbol)
            hist = stock.history(period='1d')
            if not hist.empty:
                return hist[['Close']].rename(columns={'Close': 'Price'})
            return None
        except Exception as e:
            print(f"Error fetching real-time data: {e}")
            return None