import pandas as pd
from sqlalchemy import create_engine
import pickle

engine = create_engine('postgresql://postgres:password@localhost:5432/market')
df = pd.read_sql("SELECT ts, close, volume FROM ohlcv ORDER BY ts", engine)

df['return_1h']  = df['close'].pct_change(1)
df['return_3h']  = df['close'].pct_change(3)
df['vol_change'] = df['volume'].pct_change(1)
df['target']     = (df['close'].shift(-1) > df['close']).astype(int)
df = df.dropna()

features = ['return_1h', 'return_3h', 'vol_change']

# Load saved model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Use only the test portion (last 30%)
split = int(len(df) * 0.7)
test_df = df.iloc[split:].copy()
test_df['pred'] = model.predict(test_df[features])

# Simulate: if model says "up" (1), we go long and capture that hour's return.
# If model says "down" (0), we sit out (return = 0).
test_df['actual_return'] = test_df['close'].pct_change().shift(-1)
test_df['strategy_return'] = test_df.apply(
    lambda row: row['actual_return'] if row['pred'] == 1 else 0, axis=1
)
test_df = test_df.dropna()

# Metrics
mean_r = test_df['strategy_return'].mean()
std_r  = test_df['strategy_return'].std()

sharpe = (mean_r / std_r) * (24 * 365) ** 0.5  # annualised (hourly data)

neg_returns = test_df['strategy_return'][test_df['strategy_return'] < 0]
sortino = (mean_r / neg_returns.std()) * (24 * 365) ** 0.5

cumulative = (1 + test_df['strategy_return']).cumprod().iloc[-1] - 1

print(f"Cumulative return:  {cumulative:.1%}")
print(f"Annualised Sharpe:  {sharpe:.2f}")
print(f"Annualised Sortino: {sortino:.2f}")
