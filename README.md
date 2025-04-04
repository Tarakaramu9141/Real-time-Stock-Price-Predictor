# 📈 Real-Time Stock Price Predictor

![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)
[![Open in GitHub](https://img.shields.io/badge/GitHub-View%20Source-blue)](https://github.com/yourusername/stock-predictor)

An advanced LSTM-based stock price prediction system with real-time dashboard powered by Streamlit.

## 🔥 Features

- Real-time stock data fetching from Yahoo Finance
- Deep Learning model (LSTM) for accurate price prediction
- Support for multiple stocks (AAPL, MSFT, TSLA, etc.)
- Interactive visualization with Plotly
- Auto-retraining capability
- Cloud-ready deployment

## 🚀 Live Demo

Access the live prediction dashboard directly:
[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)]
(https://real-time-stock-price-predictor-trox.streamlit.app/)

## 📊 How to Use
 
Checking Different Stocks

1. Open the sidebar by clicking the arrow in the top-left corner.
2. Enter the stock ticker symbol you want to analyze (e.g., "MSFT" for Microsoft).
3. Click "Train Model" to train a new prediction model for that stock.
4. View real-time predictions and historical data

## Popular Stock Tickers

Company	Ticker
Apple	AAPL
Microsoft	MSFT
Tesla	TSLA
Amazon	AMZN
Google	GOOGL
NVIDIA	NVDA

## 🛠️ Technical Details

Model Architecture
 1. 3-layer LSTM network with dropout (0.2)
 2. 50 epochs with early stopping
 3. 60-day lookback window

Data Pipeline
1. Fetches daily adjusted closing prices
2. Automatically handles different data formats
3. Real-time updates every 15-60 seconds (configurable)

## 🛠️ Installation

1. Clone the repository:
bash
git clone https://github.com/yourusername/stock-predictor.git
cd stock-predictor

2. Install Dependencies:
bash
pip install -r requirements.txt

3.Setup your Alpha Vanatage API key:
bash
export ALPHA_VANTAGE_API_KEY='your_api_key_here'

## 🏃‍♂️ Running the Application

1. Train the model (first time only):
bash
python -c "from src.model_trainer import StockPredictor; StockPredictor().train_model()"

2. Launch the Streamlit app:
bash
streamlit run app.py

## 📊 Model Performance
Metric	Value
MAE	$2.34
RMSE	$3.12
MAPE	1.8%

## 📚 Documentation

1. Data Pipeline: Fetches real-time data every 15 minutes.
2. Model Architecture: 3-layer LSTM with dropout (0.2).
3. Training: 50 epochs with early stopping.

## 📬 Contact
For questions or improvements

1. Email: tarakram9141@gmail.com
2. Linkedin: 
3. GitHub issues: Open Issue

## 📜 License

MIT License - See LICENSE for details


