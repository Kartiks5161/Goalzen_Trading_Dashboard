import yfinance as yf
import pandas as pd

ticker = "RELIANCE.NS"   # or any NIFTY 50
df = yf.download(ticker, period="2y", interval="1d", auto_adjust=False, progress=True)

# Normalize columns and save
df = df.reset_index()  # Date column
df.to_csv(f"{ticker.replace('.','_')}_2y_ohlcv.csv", index=False)
print("Saved:", f"{ticker.replace('.','_')}_2y_ohlcv.csv")
