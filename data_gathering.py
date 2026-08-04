from binance.spot import Spot
import pandas as pd
from datetime import datetime
import time

client = Spot()

SYMBOL = "BTCUSDT"
INTERVAL = "5m"

YEAR = 2024

start = int(datetime(YEAR, 1, 1).timestamp() * 1000)
end = int(datetime(YEAR + 1, 1, 1).timestamp() * 1000)

all_data = []

while start < end:
    candles = client.klines(
        symbol=SYMBOL,
        interval=INTERVAL,
        startTime=start,
        endTime=end,
        limit=1000
    )

    if len(candles) == 0:
        break

    all_data.extend(candles)

    # Next request starts after the last candle
    start = candles[-1][0] + 1

    print(f"Downloaded {len(all_data)} candles")

    # Be nice to the API
    time.sleep(0.1)

columns = [
    "Open Time",
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    "Close Time",
    "Quote Asset Volume",
    "Number of Trades",
    "Taker Buy Base Volume",
    "Taker Buy Quote Volume",
    "Ignore",
]

df = pd.DataFrame(all_data, columns=columns)

# Convert timestamps
df["Open Time"] = pd.to_datetime(df["Open Time"], unit="ms")
df["Close Time"] = pd.to_datetime(df["Close Time"], unit="ms")

# Convert numeric columns
numeric_cols = [
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    "Quote Asset Volume",
    "Taker Buy Base Volume",
    "Taker Buy Quote Volume",
]

df[numeric_cols] = df[numeric_cols].astype(float)

df["Number of Trades"] = df["Number of Trades"].astype(int)

# Remove any duplicates
df.drop_duplicates(subset="Open Time", inplace=True)

print(df.shape)
print(df.head())

df.to_csv("BTCUSDT_5m_2024.csv", index=False)

print("Saved to BTCUSDT_5m_2024.csv")