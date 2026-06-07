import ccxt
import pandas as pd
from sqlalchemy import create_engine

engine = create_engine('postgresql://postgres:password@localhost:5432/market')
exchange = ccxt.binance()

print("Fetching data from Binance...")
bars = exchange.fetch_ohlcv('BTC/USDT', timeframe='1h', limit=500)
df = pd.DataFrame(bars, columns=['timestamp','open','high','low','close','volume'])
df['ts'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
df['symbol'] = 'BTC/USDT'
df = df[['ts','symbol','open','high','low','close','volume']]
df.to_sql('ohlcv', engine, if_exists='replace', index=False)
print(f"Saved {len(df)} rows to the database.")
