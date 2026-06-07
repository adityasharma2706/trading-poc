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

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

split = int(len(df) * 0.7)
test_df = df.iloc[split:].copy()
test_df['pred'] = model.predict(test_df[features])
test_df['actual_return'] = test_df['close'].pct_change().shift(-1)
test_df['strategy_return'] = test_df.apply(
    lambda row: row['actual_return'] if row['pred'] == 1 else 0, axis=1)
test_df = test_df.dropna()

mean_r = test_df['strategy_return'].mean()
std_r  = test_df['strategy_return'].std()
sharpe = (mean_r / std_r) * (24 * 365) ** 0.5
neg    = test_df['strategy_return'][test_df['strategy_return'] < 0]
sortino   = (mean_r / neg.std()) * (24 * 365) ** 0.5
cumulative = (1 + test_df['strategy_return']).cumprod().iloc[-1] - 1

print(f"Cumulative return:  {cumulative:.1%}")
print(f"Annualised Sharpe:  {sharpe:.2f}")
print(f"Annualised Sortino: {sortino:.2f}")

results = []
window, step = 200, 50
for start in range(0, len(df) - window - step, step):
    train = df.iloc[start:start+window]
    test  = df.iloc[start+window:start+window+step]
    m = RandomForestClassifier(n_estimators=50, random_state=42)
    m.fit(train[features], train['target'])
    results.append(accuracy_score(test['target'], m.predict(test[features])))
print(f"Walk-forward mean:  {sum(results)/len(results):.1%}")
print(f"Best fold: {max(results):.1%}  Worst fold: {min(results):.1%}")
