import redis
import ccxt
import pandas as pd
from sqlalchemy import create_engine
from sklearn.ensemble import RandomForestClassifier
import pickle
import time

# Connections
r = redis.Redis(host='localhost', port=6379, decode_responses=True)
engine = create_engine('postgresql://postgres:password@localhost:5432/market')
exchange = ccxt.binance()

def fetch_and_store():
    print("[1/3] Fetching market data...")
    bars = exchange.fetch_ohlcv('BTC/USDT', timeframe='1h', limit=500)
    df = pd.DataFrame(bars, columns=['timestamp','open','high','low','close','volume'])
    df['ts'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    df['symbol'] = 'BTC/USDT'
    df = df[['ts','symbol','open','high','low','close','volume']]
    df.to_sql('ohlcv', engine, if_exists='replace', index=False)
    print(f"    Stored {len(df)} rows.")
    # Publish event so any subscriber knows new data is ready
    r.publish('market_data', 'BTC/USDT:ready')

def train():
    print("[2/3] Training model...")
    df = pd.read_sql("SELECT ts, close, volume FROM ohlcv ORDER BY ts", engine)
    df['return_1h']  = df['close'].pct_change(1)
    df['return_3h']  = df['close'].pct_change(3)
    df['vol_change'] = df['volume'].pct_change(1)
    df['target']     = (df['close'].shift(-1) > df['close']).astype(int)
    df = df.dropna()

    features = ['return_1h', 'return_3h', 'vol_change']
    split = int(len(df) * 0.7)
    X_train = df[features].iloc[:split]
    y_train = df['target'].iloc[:split]

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)
    print("    Model trained and saved.")
    r.publish('model_ready', 'model.pkl:ready')

def generate_signal():
    print("[3/3] Generating trading signal...")
    df = pd.read_sql("SELECT close, volume FROM ohlcv ORDER BY ts DESC LIMIT 10", engine)
    df = df.iloc[::-1].reset_index(drop=True)  # oldest first

    df['return_1h']  = df['close'].pct_change(1)
    df['return_3h']  = df['close'].pct_change(3)
    df['vol_change'] = df['volume'].pct_change(1)
    df = df.dropna()

    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)

    latest = df[['return_1h','return_3h','vol_change']].iloc[[-1]]
    pred = model.predict(latest)[0]
    signal = 'BUY' if pred == 1 else 'HOLD'

    # Publish signal so an order executor could act on it
    r.publish('signals', f'BTC/USDT:{signal}')
    print(f"    Signal published: {signal}")
    return signal

if __name__ == '__main__':
    fetch_and_store()
    train()
    signal = generate_signal()
    print(f"\nDone. Final signal: {signal}")
