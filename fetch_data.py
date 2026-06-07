import ccxt
import pandas as pd
from sqlalchemy import create_engine

# Connect to the database
engine = create_engine('postgresql://postgres:password@localhost:5432/market')

# Connect to Binance (no account needed for public data)
exchange = ccxt.binance()

# Fetch the last 500 hourly candles for BTC/USDT
print("Fetching data from Binance...")
bars = exchange.fetch_ohlcv('BTC/USDT', timeframe='1h', limit=500)

# Turn it into a DataFrame (like a table in memory)
df = pd.DataFrame(bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
df['ts'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
df['symbol'] = 'BTC/USDT'
df = df[['ts', 'symbol', 'open', 'high', 'low', 'close', 'volume']]

# Write to TimescaleDB
df.to_sql('ohlcv', engine, if_exists='append', index=False)
print(f"Saved {len(df)} rows to the database.")
