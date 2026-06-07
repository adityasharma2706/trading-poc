import redis
import ccxt
import pandas as pd
from sqlalchemy import create_engine
from sklearn.ensemble import RandomForestClassifier
import pickle

r = redis.Redis(host='localhost', port=6379, decode_responses=True)
engine = create_engine('postgresql://postgres:password@localhost:5432/market')
exchange = ccxt.binance()

def compute_rsi(series, period=14):
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss
    return 100 - (100 / (1 + rs))

def fetch_and_store():
    print("[1/3] Fetching market data...")
    bars = exchange.fetch_ohlcv('BTC/USDT', timeframe='1h', limit=500)
    df = pd.DataFrame(bars, columns=['timestamp','open','high','low','close','volume'])
    df['ts'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    df['symbol'] = 'BTC/USDT'
    df = df[['ts','symbol','open','high','low','close','volume']]
    df.to_sql('ohlcv', engine, if_exists='replace', index=False)
    print(f"    Stored {len(df)} rows.")
    r.publish('market_data', 'BTC/USDT:ready')

def train():
    print("[2/3] Training model...")
    df = pd.read_sql("SELECT ts, close, volume FROM ohlcv ORDER BY ts", engine)
    df['return_1h']  = df['close'].pct_change(1)
    df['return_3h']  = df['close'].pct_change(3)
    df['vol_change'] = df['volume'].pct_change(1)
    df['sma_10']     = df['close'].rolling(10).mean()
    df['sma_50']     = df['close'].rolling(50).mean()
    df['rsi']        = compute_rsi(df['close'], 14)
    df['bb_width']   = (df['close'].rolling(20).std() * 2) / df['close'].rolling(20).mean()
    df['target']     = (df['close'].shift(-1) > df['close']).astype(int)
    df = df.dropna()
    features = ['return_1h','return_3h','vol_change','sma_10','sma_50','rsi','bb_width']
    split = int(len(df) * 0.7)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(df[features].iloc[:split], df['target'].iloc[:split])
    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)
    print("    Model trained and saved.")
    r.publish('model_ready', 'model.pkl:ready')

def generate_signal():
    print("[3/3] Generating trading signal...")
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
    latest = df[features].iloc[[-1]]
    prob   = model.predict_proba(latest)[0][1]
    if prob > 0.60:
        signal = 'BUY'
    elif prob < 0.40:
        signal = 'SELL'
    else:
        signal = 'HOLD'
    r.publish('signals', f'BTC/USDT:{signal}')
    print(f"    Confidence: {prob:.0%}  →  Signal: {signal}")
    return signal

if __name__ == '__main__':
    fetch_and_store()
    train()
    signal = generate_signal()
    print(f"\nDone. Final signal: {signal}")
