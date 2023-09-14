import yfinance as yf
import plotly.graph_objects as go
import plotly.subplots as sp

# Define the ticker symbol for the stock you want to fetch data for
ticker_symbol = "AAPL"

# Fetch the data from Yahoo Finance
data = yf.download(ticker_symbol, start="2022-01-01", end="2022-12-31")

# Extract the required data for each indicator
price_data = data[['Close']]

# Assuming you have pandas DataFrames or arrays for each indicator: moving_average_data, rsi_data, and bollinger_data

# Create a subplot with 2 rows and 2 columns
fig = sp.make_subplots(rows=2, cols=2, subplot_titles=("Price", "Moving Average", "RSI", "Bollinger Bands"))

# Add Price chart
fig.add_trace(go.Scatter(x=price_data.index, y=price_data['Close'], name='Price'), row=1, col=1)

# Add Moving Average chart
fig.add_trace(go.Scatter(x=moving_average_data.index, y=moving_average_data['50-day MA'], name='Moving Average'), row=1, col=2)

# Add RSI chart
fig.add_trace(go.Scatter(x=rsi_data.index, y=rsi_data['RSI'], name='RSI'), row=2, col=1)

# Add Bollinger Bands chart
fig.add_trace(go.Scatter(x=bollinger_data.index, y=bollinger_data['Bollinger Mean'], name='Bollinger Mean'), row=2, col=2)
fig.add_trace(go.Scatter(x=bollinger_data.index, y=bollinger_data['Bollinger Upper'], name='Bollinger Upper'), row=2, col=2)
fig.add_trace(go.Scatter(x=bollinger_data.index, y=bollinger_data['Bollinger Lower'], name='Bollinger Lower'), row=2, col=2)

# Update x-axis and y-axis labels
fig.update_xaxes(title_text="Date", row=1, col=1)
fig.update_xaxes(title_text="Date", row=1, col=2)
fig.update_xaxes(title_text="Date", row=2, col=1)
fig.update_xaxes(title_text="Date", row=2, col=2)

fig.update_yaxes(title_text="Price", row=1, col=1)
fig.update_yaxes(title_text="Moving Average", row=1, col=2)
fig.update_yaxes(title_text="RSI", row=2, col=1)
fig.update_yaxes(title_text="Bollinger Bands", row=2, col=2)

# Update layout properties
fig.update_layout(
    title_text=f"Financial Dashboard for {ticker_symbol}",
    height=600,
    showlegend=True,
    legend=dict(
        x=0,
        y=1,
        traceorder="normal",
        font=dict(
            family="sans-serif",
            size=12,
            color="black"
        ),
    ),
)

# Display the dashboard
fig.show()
