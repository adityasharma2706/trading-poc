import pandas as pd
from sqlalchemy import create_engine
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

engine = create_engine('postgresql://postgres:password@localhost:5432/market')
df = pd.read_sql("SELECT ts, close, volume FROM ohlcv ORDER BY ts", engine)

def compute_rsi(series, period=14):
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss
    return 100 - (100 / (1 + rs))

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
X_train, X_test = df[features].iloc[:split], df[features].iloc[split:]
y_train, y_test = df['target'].iloc[:split], df['target'].iloc[split:]

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
preds = model.predict(X_test)
print(f"Hit rate on test data: {accuracy_score(y_test, preds):.1%}")

with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
print("Model saved to model.pkl")
