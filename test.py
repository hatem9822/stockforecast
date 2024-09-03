import os
import logging
import pandas as pd
import streamlit as st
import yfinance as yf
import matplotlib.pyplot as plt
import ta
import pickle
from sklearn.preprocessing import MinMaxScaler

from neuralforecast.core import NeuralForecast
from neuralforecast.models import iTransformer, TSMixer, TimeMixer, NHITS
from utilsforecast.losses import mae, mse, smape
from utilsforecast.evaluation import evaluate

# Set logging level to suppress TensorBoard logging
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)

# Function to load stock data and calculate selected indicators
def load_stock_data_with_indicators(symbol, start_date, end_date, indicators):
    stock_data = yf.download(symbol, start=start_date, end=end_date)
    
    # Calculate selected indicators
    if 'rsi' in indicators:
        stock_data['rsi'] = ta.momentum.RSIIndicator(stock_data['Close'], window=14).rsi()
    if 'ma_50' in indicators:
        stock_data['ma_50'] = ta.trend.SMAIndicator(stock_data['Close'], window=50).sma_indicator()
    if 'ma_200' in indicators:
        stock_data['ma_200'] = ta.trend.SMAIndicator(stock_data['Close'], window=200).sma_indicator()
    if 'ema_50' in indicators:
        stock_data['ema_50'] = ta.trend.EMAIndicator(stock_data['Close'], window=50).ema_indicator()
    if 'stoch_k' in indicators:
        stoch = ta.momentum.StochasticOscillator(stock_data['High'], stock_data['Low'], stock_data['Close'], window=14)
        stock_data['stoch_k'] = stoch.stoch()  # %K line only
    if 'macd' in indicators:
        macd = ta.trend.MACD(stock_data['Close'], window_slow=26, window_fast=12, window_sign=9)
        stock_data['macd'] = macd.macd()
        stock_data['macd_signal'] = macd.macd_signal()
    
    stock_data.dropna(inplace=True)
    
    # Update selected columns to include stoch_k and macd indicators
    selected_columns = ['Close'] + [ind for ind in indicators if ind != 'macd']
    macd_columns = ['macd', 'macd_signal'] if 'macd' in indicators else []
    
    stock_data = stock_data[['Close'] + selected_columns + macd_columns].reset_index()
    stock_data.columns = ['ds', 'y'] + selected_columns + macd_columns
    stock_data['unique_id'] = symbol

    # Create a separate scaler for the 'y' column
    y_scaler = MinMaxScaler(feature_range=(0, 1))
    stock_data['y'] = y_scaler.fit_transform(stock_data[['y']])
    
    # Scale the rest of the indicators using another scaler (optional)
    scaler = MinMaxScaler(feature_range=(0, 1))
    stock_data[selected_columns + macd_columns] = scaler.fit_transform(stock_data[selected_columns + macd_columns])
    
    return stock_data, y_scaler  # Return both stock_data and y_scaler

# Function to check if a model file already exists
def model_exists(filename):
    return os.path.exists(filename)

# Function to save the model
def save_model(model, filename):
    with open(filename, 'wb') as file:
        pickle.dump(model, file)

# Function to load a saved model
def load_model(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)

# Streamlit app
st.title("Stock Price Forecasting App")

# Sidebar inputs
st.sidebar.header("Input Parameters")
symbol = st.sidebar.text_input("Stock Symbol", "AAPL")
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2000-01-31"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("2018-08-18"))
display_days = st.sidebar.slider("Number of Days to Predict", 1, 30, 7)  
indicators = st.sidebar.multiselect("Indicators to Include", [
    'rsi', 'ma_50', 'ma_200', 'ema_50', 
     'stoch_k', 'macd'], 
    default=['rsi', 'ma_50'])


horizon = 30
input_size = 60

# Load data and y_scaler
stock_data, y_scaler = load_stock_data_with_indicators(symbol, start_date, end_date, indicators)

# Prepare training and test sets
test_df = stock_data.groupby('unique_id').tail(horizon)
train_df = stock_data.drop(test_df.index).reset_index(drop=True)


model_choice = st.sidebar.selectbox("Model", ["iTransformer", "TSMixer", "TimeMixer", "NHITS"])

# Generate a unique filename based on selected parameters
indicator_str = '_'.join(indicators)
model_filename = f"{symbol}_{model_choice}_{start_date}_{end_date}_{horizon}days_{indicator_str}.pkl"


if st.button("Start Training"):
    # Load existing model if it exists, otherwise train a new one
    if model_exists(model_filename):
        st.write("Loading existing model...")
        nf = load_model(model_filename)
    else:
        st.write("Training a new model...")
        

        if model_choice == "iTransformer":
            model = iTransformer(input_size=96, 
                                h=horizon, n_series=1,
                                scaler_type='identity', 
                                max_steps=1000, 
                                early_stop_patience_steps=3)
        
        elif model_choice == "TSMixer":
            model = TSMixer(input_size=input_size, 
                            h=horizon, n_series=1,
                            scaler_type='identity',
                            max_steps=1000, 
                            early_stop_patience_steps=3)
        
        elif model_choice == "TimeMixer":
            model = TimeMixer(input_size=60, 
                              h=horizon, 
                              n_series=1, 
                              scaler_type='identity',
                                early_stop_patience_steps=3)
        
        elif model_choice == "NHITS":
            model = NHITS(input_size=input_size,
                    h=horizon,
                    scaler_type='identity',
                    max_steps=1000,
                    early_stop_patience_steps=3)

        nf = NeuralForecast(models=[model], freq='D')
        
    
        nf.fit(train_df, val_size=horizon)
        save_model(nf, model_filename)

 
    preds = nf.predict().reset_index()

    
    preds = preds[preds['ds'].isin(test_df['ds'])].reset_index(drop=True)


    test_df = test_df.iloc[:len(preds)].reset_index(drop=True)

    # Merge predictions with aligned test data
    test_df = pd.merge(test_df, preds, on=['ds', 'unique_id'], how='left')

    # Rescale the `y` and predicted columns back to original scale using y_scaler
    test_df['y'] = y_scaler.inverse_transform(test_df[['y']])
    test_df[f'{model_choice}'] = y_scaler.inverse_transform(test_df[[f'{model_choice}']])

   
    test_df = test_df.iloc[:display_days].reset_index(drop=True)

    
    evaluation = evaluate(test_df, metrics=[mae, mse, smape], models=[model_choice], target_col="y")

    
    st.write("## Model Evaluation")
    st.table(evaluation)


    st.write(f"### {symbol} Stock Price Forecast with {model_choice} (Last 1 Year + Forecast)")
    plt.figure(figsize=(12, 6))

   
    train_df['y'] = y_scaler.inverse_transform(train_df[['y']])
    plt.plot(train_df['ds'].tail(365), train_df['y'].tail(365), label='Actual (Train)', color='black')

    plt.plot(test_df['ds'], test_df['y'], label='Actual (Test)', color='blue')
    plt.plot(test_df['ds'], test_df[f'{model_choice}'], label=f'Forecast ({model_choice})', color='red')

    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    st.pyplot(plt)

    
    st.write(f"### {symbol} Stock Price Forecast with {model_choice} (Displaying {display_days} Days)")
    plt.figure(figsize=(12, 6))
    plt.plot(test_df['ds'], test_df['y'], label='Actual (Test)', color='blue')
    plt.plot(test_df['ds'], test_df[f'{model_choice}'], label=f'Forecast ({model_choice})', color='red')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    st.pyplot(plt)
