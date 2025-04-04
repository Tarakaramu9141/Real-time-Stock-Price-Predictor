import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import joblib

class StockPricePredictor:
    def __init__(self, model_path='models/lstm_stock_model.h5', scaler_path='models/scaler.pkl'):
        self.model = load_model(model_path)
        self.scaler = joblib.load(scaler_path)
        self.lookback = 60
        
    def predict_future(self, data, days=30):
        """Predict future stock prices"""
        # Prepare data
        scaled_data = self.scaler.transform(data.values.reshape(-1, 1))
        
        # Create input sequence
        input_seq = scaled_data[-self.lookback:]
        predictions = []
        
        for _ in range(days):
            # Reshape for LSTM
            x = input_seq[-self.lookback:].reshape(1, self.lookback, 1)
            
            # Predict next day
            pred = self.model.predict(x)
            predictions.append(pred[0, 0])
            
            # Update input sequence
            input_seq = np.append(input_seq, pred)
        
        # Inverse transform predictions
        predictions = self.scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
        
        # Create future dates
        last_date = data.index[-1]
        future_dates = pd.date_range(start=last_date, periods=days+1)[1:]
        
        return pd.Series(predictions.flatten(), index=future_dates)