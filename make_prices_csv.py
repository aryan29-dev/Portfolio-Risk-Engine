import pandas as pd
import yfinance as yf
import os

tickers = ["AAPL", "MSFT", "NVDA", "TSLA", "VOO", "AMZN", "GOOGL", "META", "BRK-B", "JPM"]

start_date = "2023-01-01"

ticker_data = yf.download(tickers=tickers, start=start_date,interval="1d", group_by="ticker", auto_adjust=False)

close_prices = pd.DataFrame()

for ticker in tickers:
    close_prices[ticker] = ticker_data[ticker]["Close"]

close_prices = close_prices.reset_index()

os.makedirs("data", exist_ok=True)

close_prices.to_csv("data/prices.csv", index=False)

print("Saved: data/prices.csv")
print("Columns:", list(close_prices.columns))
