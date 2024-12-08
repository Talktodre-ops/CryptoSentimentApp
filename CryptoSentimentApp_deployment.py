import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import requests

# Function to fetch real-time cryptocurrency price
def fetch_live_price(crypto_id, currency='usd'):
    """
    Fetch live cryptocurrency price from CoinGecko API.
    
    Args:
    - crypto_id (str): The CoinGecko ID of the cryptocurrency (e.g., 'bitcoin').
    - currency (str): The fiat currency to fetch the price in (default 'usd').

    Returns:
    - float: The live price or None if an error occurs.
    """
    url = "https://api.coingecko.com/api/v3/simple/price"
    params = {"ids": crypto_id, "vs_currencies": currency}
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        return response.json()[crypto_id][currency]
    except requests.exceptions.RequestException:
        return None

# Load the trained model
model = joblib.load('best_rf_model.pkl')

# Load the predicted data
predicted_data = pd.read_csv('predicted_crypto_prices.csv')

# Convert date column to datetime
predicted_data['date'] = pd.to_datetime(predicted_data['date'])

# Handle missing values in predicted_data
predicted_data.fillna(method='ffill', inplace=True)

# Set page configuration
st.set_page_config(page_title='Cryptocurrency Price Prediction', layout='wide')

# Title and description
st.title('Cryptocurrency Price Prediction')
st.markdown("""
This app visualizes the predicted cryptocurrency prices based on sentiment analysis.
Use the interactive features to explore the data and gain insights.
""")

# Sidebar for user inputs
st.sidebar.header('User Input Features')

# Fetch and display real-time price
crypto_id = "bitcoin"  # CoinGecko ID for Bitcoin
live_price = fetch_live_price(crypto_id)

if live_price:
    st.sidebar.markdown(f"**Live {crypto_id.capitalize()} Price**: ${live_price:,.2f}")
else:
    st.sidebar.error("Error fetching live price data.")

date_range = st.sidebar.date_input(
    'Select Date Range', 
    [predicted_data['date'].min(), predicted_data['date'].max()]
)

# Ensure sentiment range is valid
sentiment_min = float(predicted_data['average_sentiment'].min())
sentiment_max = float(predicted_data['average_sentiment'].max())
sentiment_range = st.sidebar.slider(
    'Select Sentiment Range', 
    sentiment_min, sentiment_max, (sentiment_min, sentiment_max)
)

# Filter data based on user input
# Display available date range
min_date = predicted_data['date'].min()
max_date = predicted_data['date'].max()
st.sidebar.markdown(f"**Available Data Range**: {min_date.date()} to {max_date.date()}")

# Validate and adjust date range
if len(date_range) == 2:
    start_date, end_date = date_range
    start_date = max(pd.to_datetime(start_date), min_date)
    end_date = min(pd.to_datetime(end_date), max_date)
else:
    start_date, end_date = min_date, max_date

# Filter data
filtered_data = predicted_data[
    (predicted_data['date'] >= start_date) &
    (predicted_data['date'] <= end_date) &
    (predicted_data['average_sentiment'] >= sentiment_range[0]) &
    (predicted_data['average_sentiment'] <= sentiment_range[1])
]

# Handle empty data
if filtered_data.empty:
    st.warning("No data available for the selected filters. Please adjust the date or sentiment range.")
    # Preview available data
    st.write("Here’s a preview of the available data:")
    st.write(predicted_data[['date', 'price', 'predicted_price', 'average_sentiment']].head())
else:
    # Calculate moving averages
    filtered_data['short_ma'] = filtered_data['price'].rolling(window=10).mean()  # Short-term MA
    filtered_data['long_ma'] = filtered_data['price'].rolling(window=30).mean()  # Long-term MA

    # Create Line Chart
    st.subheader('Line Chart of Cryptocurrency Prices')
    fig = go.Figure()

    # Add Actual Price Line
    fig.add_trace(go.Scatter(
        x=filtered_data['date'],
        y=filtered_data['price'],
        mode='lines',
        line=dict(color='blue', width=2),
        name='Actual Price'
    ))

    # Add Predicted Price Line
    fig.add_trace(go.Scatter(
        x=filtered_data['date'],
        y=filtered_data['predicted_price'],
        mode='lines',
        line=dict(color='orange', width=2),
        name='Predicted Price'
    ))

    # Add Short-Term Moving Average
    fig.add_trace(go.Scatter(
        x=filtered_data['date'],
        y=filtered_data['short_ma'],
        mode='lines',
        line=dict(color='magenta', width=2),
        name='10-Day MA'
    ))

    # Add Long-Term Moving Average
    fig.add_trace(go.Scatter(
        x=filtered_data['date'],
        y=filtered_data['long_ma'],
        mode='lines',
        line=dict(color='cyan', width=2),
        name='30-Day MA'
    ))

    # Layout Configuration
    fig.update_layout(
        title='Cryptocurrency Price Prediction with Moving Averages',
        xaxis_title='Date',
        yaxis_title='Price',
        template='plotly_white'
    )

    # Render in Streamlit
    st.plotly_chart(fig, use_container_width=True)

    # Additional visualizations
    st.subheader('Price Distribution')
    fig2 = px.histogram(filtered_data, x='price', nbins=50, title='Price Distribution')
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader('Sentiment vs. Price')
    fig3 = px.scatter(filtered_data, x='average_sentiment', y='price', color='price', title='Sentiment vs. Price')
    st.plotly_chart(fig3, use_container_width=True)

    st.subheader('Correlation Heatmap')
    fig4, ax = plt.subplots(figsize=(10, 8))
    correlation_matrix = filtered_data.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig4)

# Model Performance Metrics
st.subheader('Model Performance Metrics')
st.markdown("""
- **RMSE (Root Mean Squared Error)**: 523.63
- **MAE (Mean Absolute Error)**: 396.58
- **R-squared**: 0.997
""")

# Future Price Prediction
st.subheader('Future Price Prediction')
st.markdown("""
Enter the required data to get future price predictions.
""")

with st.form(key='prediction_form'):
    price_lag_1 = st.number_input('Price Lag 1 (or use live price)', value=live_price if live_price else 0.0)
    price_roll_mean_30 = st.number_input('Price Rolling Mean 30', value=0.0)
    price_ma_7 = st.number_input('Price Moving Average 7', value=0.0)
    price_roll_mean_7 = st.number_input('Price Rolling Mean 7', value=0.0)
    price_ma_30 = st.number_input('Price Moving Average 30', value=0.0)
    submit_button = st.form_submit_button(label='Predict')

if submit_button:
    if price_lag_1 == 0.0 or price_roll_mean_30 == 0.0:
        st.error("Please provide valid inputs for all required fields.")
    else:
        # Create a DataFrame with the input data
        input_data = pd.DataFrame({
            'price_lag_1': [price_lag_1],
            'price_roll_mean_30': [price_roll_mean_30],
            'price_ma_7': [price_ma_7],
            'price_roll_mean_7': [price_roll_mean_7],
            'price_ma_30': [price_ma_30]
        })

        # Make predictions
        predicted_price = model.predict(input_data)

        # Display the predicted price
        st.success(f"Predicted Price: ${predicted_price[0]:,.2f}")
