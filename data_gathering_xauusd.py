import dukascopy_python
from dukascopy_python.instruments import INSTRUMENT_FX_METALS_XAU_USD
import pandas as pd
from datetime import datetime

# Dukascopy gives real XAU/USD (spot gold vs USD) data, free, no API key,
# unlike Binance which only offers gold-backed crypto tokens (PAXG/XAUT)
# as an imperfect proxy.
YEAR = 2026

start = datetime(YEAR, 1, 1)
end = datetime(YEAR + 1, 1, 1)

print(f"Downloading XAU/USD 5m candles for {YEAR}...")

df = dukascopy_python.fetch(
    instrument=INSTRUMENT_FX_METALS_XAU_USD,
    interval=dukascopy_python.INTERVAL_MIN_5,
    offer_side=dukascopy_python.OFFER_SIDE_BID,  # BID or ASK; BID is standard for OHLC
    start=start,
    end=end,
)

# The library returns a DataFrame indexed by timestamp with
# columns: open, high, low, close, volume
df = df.reset_index()
df.rename(columns={
    "timestamp": "Open Time",
    "open": "Open",
    "high": "High",
    "low": "Low",
    "close": "Close",
    "volume": "Volume",
}, inplace=True)

# Remove any duplicates
df.drop_duplicates(subset="Open Time", inplace=True)

print(df.shape)
print(df.head())

df.to_csv(f"XAUUSD_5m_{YEAR}.csv", index=False)

print(f"Saved to XAUUSD_5m_{YEAR}.csv")