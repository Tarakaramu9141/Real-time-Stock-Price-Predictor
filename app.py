import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from src.data_fetcher import DataFetcher
from src.model_trainer import StockPredictor
from src.predictor import StockPricePredictor
import time
import os
import uuid

# Set page config
st.set_page_config(
    page_title="Real-Time Stock Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        background-color: #0E1117;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
    }
    .stTextInput>div>div>input {
        background-color: #0E1117;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

# Title
st.title('📈 Real-Time Stock Price Predictor')
st.markdown("Predict future stock prices using LSTM deep learning model.")

# Sidebar
st.sidebar.header('Settings')
ticker = st.sidebar.text_input('Stock Ticker (e.g., AAPL)', 'AAPL')
days_to_predict = st.sidebar.slider('Days to Predict', 1, 60, 30)
refresh_rate = st.sidebar.slider('Refresh Rate (seconds)', 5, 60, 15)

# Initialize data fetcher
fetcher = DataFetcher()

# Initialize predictor with error handling
predictor = None
if os.path.exists('models/lstm_stock_model.h5'):
    try:
        predictor = StockPricePredictor()
    except Exception as e:
        st.sidebar.warning(f"Model loading warning: {str(e)}")

# Function to train model
def train_model():
    with st.spinner('Training model... This may take several minutes.'):
        try:
            data = fetcher.fetch_historical_data(ticker)
            if data is None or data.empty:
                st.error("Failed to fetch training data. Please check your internet connection.")
                return False
                
            trainer = StockPredictor()
            trainer.train_model()
            st.success('Model trained successfully!')
            return True
        except Exception as e:
            st.error(f"Training failed: {str(e)}")
            return False

# Train model button
if st.sidebar.button('Train Model'):
    if train_model():
        try:
            predictor = StockPricePredictor()
        except Exception as e:
            st.error(f"Failed to load predictor after training: {str(e)}")

# Main app container
placeholder = st.empty()

# Session state for maintaining unique keys
if 'chart_keys' not in st.session_state:
    st.session_state.chart_keys = {
        'historical': str(uuid.uuid4()),
        'prediction': str(uuid.uuid4()),
        'table': str(uuid.uuid4())
    }

while True:
    with placeholder.container():
        try:
            # Fetch data
            historical_data = fetcher.fetch_historical_data(ticker)
            real_time_data = fetcher.fetch_real_time_data(ticker)
            
            if historical_data is not None and not historical_data.empty:
                # Standardize column names
                if 'Price' not in historical_data.columns:
                    if 'Close' in historical_data.columns:
                        historical_data = historical_data[['Close']].rename(columns={'Close': 'Price'})
                
                # Display current price
                if real_time_data is not None and not real_time_data.empty:
                    current_price = real_time_data.iloc[-1]['Price'] if isinstance(real_time_data, pd.DataFrame) else real_time_data['Price'][0]
                    st.metric(label=f"Current {ticker} Price", 
                            value=f"${float(current_price):.2f}")
                
                # Plot historical data with persistent key
                st.subheader('Historical Price Data')
                fig_hist = go.Figure()
                fig_hist.add_trace(go.Scatter(
                    x=historical_data.index,
                    y=historical_data['Price'],
                    name='Historical Price',
                    line=dict(color='#1f77b4')
                ))
                fig_hist.update_layout(
                    xaxis_title='Date',
                    yaxis_title='Price ($)',
                    template='plotly_dark'
                )
                st.plotly_chart(fig_hist, 
                              use_container_width=True, 
                              key=st.session_state.chart_keys['historical'])
                
                # Make predictions if model is available
                if predictor is not None:
                    st.subheader('Future Price Prediction')
                    try:
                        predictions = predictor.predict_future(
                            historical_data['Price'], 
                            days=days_to_predict
                        )
                        
                        # Plot predictions with persistent key
                        fig_pred = go.Figure()
                        fig_pred.add_trace(go.Scatter(
                            x=historical_data.index[-60:],
                            y=historical_data['Price'][-60:],
                            name='Recent History',
                            line=dict(color='#1f77b4')
                        ))
                        fig_pred.add_trace(go.Scatter(
                            x=predictions.index,
                            y=predictions.values,
                            name='Predicted Price',
                            line=dict(color='#ff7f0e', dash='dot')
                        ))
                        fig_pred.update_layout(
                            xaxis_title='Date',
                            yaxis_title='Price ($)',
                            template='plotly_dark'
                        )
                        st.plotly_chart(fig_pred, 
                                      use_container_width=True, 
                                      key=st.session_state.chart_keys['prediction'])
                        
                        # Show prediction table with persistent key
                        st.write("Predicted Prices:")
                        st.dataframe(
                            predictions.rename('Predicted Price')
                            .to_frame()
                            .style.format("${:.2f}"),
                            key=st.session_state.chart_keys['table']
                        )
                    
                    except Exception as e:
                        st.error(f"Prediction failed: {str(e)}")
            else:
                st.warning("No historical data available. Please train the model first.")
        
        except Exception as e:
            st.error(f"Application error: {str(e)}")
    
    time.sleep(refresh_rate)