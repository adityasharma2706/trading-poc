import csv, os, pickle, pandas as pd
from sqlalchemy import create_engine
from datetime import datetime, timezone

def compute_rsi(series, period=14):
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    return 100 - (100 / (1 + gain/loss))

engine = create_engine('postgresql://postgres:password@localhost:5432/market')
df = pd.read_sql("SELECT close, volume FROM ohlcv ORDER BY ts DESC LIMIT 60", engine)
df = df.iloc[::-1].reset_index(drop=True)
df['return_1h']  = df['close'].pct_change(1)
df['return_3h']  = df['close'].pct_change(3)
df['vol_change'] = df['volume'].pct_change(1)
df['sma_10']     = df['close'].rolling(10).mean()
df['sma_50']     = df['close'].rolling(50).mean()
df['rsi']        = compute_rsi(df['close'], 14)
df['bb_width']   = (df['close'].rolling(20).std() * 2) / df['close'].rolling(20).mean()
df = df.dropna()

features = ['return_1h','return_3h','vol_change','sma_10','sma_50','rsi','bb_width']
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

prob   = model.predict_proba(df[features].iloc[[-1]])[0][1]
signal = 'BUY' if prob > 0.60 else ('SELL' if prob < 0.40 else 'HOLD')
price  = df['close'].iloc[-1]
ts     = datetime.now(timezone.utc).isoformat()

log_exists = os.path.exists('paper_trades.csv')
with open('paper_trades.csv', 'a', newline='') as f:
    w = csv.writer(f)
    if not log_exists:
        w.writerow(['timestamp','symbol','signal','price','confidence'])
    w.writerow([ts, 'BTC/USDT', signal, price, f"{prob:.2%}"])

print(f"{ts}  {signal}  @ ${price:,.2f}  (confidence: {prob:.0%})")
print("Logged to paper_trades.csv")
