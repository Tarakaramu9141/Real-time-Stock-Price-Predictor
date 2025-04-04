import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import plotly.graph_objects as go

def calculate_metrics(actual, predicted):
    """Calculate evaluation metrics"""
    mae = mean_absolute_error(actual, predicted)
    mse = mean_squared_error(actual, predicted)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    return {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'MAPE': mape
    }

def plot_predictions(history, predictions, title='Stock Price Prediction'):
    """Create interactive prediction plot"""
    fig = go.Figure()
    
    # Historical data
    fig.add_trace(go.Scatter(
        x=history.index,
        y=history.values,
        name='Historical Prices',
        line=dict(color='#1f77b4')
    ))
    
    # Predictions
    fig.add_trace(go.Scatter(
        x=predictions.index,
        y=predictions.values,
        name='Predicted Prices',
        line=dict(color='#ff7f0e', dash='dot')
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Date',
        yaxis_title='Price ($)',
        template='plotly_dark',
        hovermode='x unified'
    )
    
    return fig

def prepare_training_data(data, lookback=60):
    """Prepare data for LSTM training"""
    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i-lookback:i])
        y.append(data[i])
    return np.array(X), np.array(y)