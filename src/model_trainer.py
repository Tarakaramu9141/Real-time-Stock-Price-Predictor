import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import os
import joblib

class StockPredictor:
    def __init__(self, data_path='data/stock_data.csv'):
        self.data_path = data_path
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = None
        self.lookback = 60  # Number of days to look back for prediction
        
    def load_data(self):
        """Load and preprocess data with multiple fallback options"""
        try:
            if not os.path.exists(self.data_path):
                raise FileNotFoundError(f"Data file not found at {self.data_path}")
            df=pd.read_csv(self.data_path,index_col='Date', parse_dates=True)
            if 'Price' not in df.columns:
                raise ValueError("Price column not found in data")
            return df[['Price']]
        except Exception as e:
            print(f"Error Loading data:{e}")
            return None
            
    
    def create_dataset(self, dataset):
        """Create time series dataset for LSTM"""
        X, y = [], []
        for i in range(self.lookback, len(dataset)):
            X.append(dataset[i-self.lookback:i, 0])
            y.append(dataset[i, 0])
        return np.array(X), np.array(y)
    
    def train_model(self):
        """Train LSTM model"""
        # Load and prepare data
        df = self.load_data()
        if df is None:
            raise ValueError("Failed to load data for training")
            
        scaled_data = self.scaler.fit_transform(df.values)
        
        # Create training data
        train_size = int(len(scaled_data) * 0.8)
        train_data = scaled_data[:train_size]
        X_train, y_train = self.create_dataset(train_data)
        
        # Reshape for LSTM [samples, timesteps, features]
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        
        # Build LSTM model
        self.model = Sequential([
            LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
            Dropout(0.2),
            LSTM(units=50, return_sequences=True),
            Dropout(0.2),
            LSTM(units=50),
            Dropout(0.2),
            Dense(units=1)
        ])
        
        self.model.compile(optimizer='adam', loss='mean_squared_error')
        
        # Train model
        early_stop = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
        self.model.fit(X_train, y_train, epochs=50, batch_size=32, callbacks=[early_stop])
        
        # Save model and scaler
        os.makedirs('models', exist_ok=True)
        self.model.save('models/lstm_stock_model.h5')
        joblib.dump(self.scaler, 'models/scaler.pkl')
        
        return self.model