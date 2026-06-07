import pandas as pd
from sqlalchemy import create_engine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle

engine = create_engine('postgresql://postgres:password@localhost:5432/market')

# Load data from DB
df = pd.read_sql("SELECT ts, close, volume FROM ohlcv ORDER BY ts", engine)

# Feature engineering: what signals do we give the model?
df['return_1h']  = df['close'].pct_change(1)   # % change last 1 hour
df['return_3h']  = df['close'].pct_change(3)   # % change last 3 hours
df['vol_change'] = df['volume'].pct_change(1)  # volume momentum

# Label: did price go UP in the next hour? 1 = yes, 0 = no
df['target'] = (df['close'].shift(-1) > df['close']).astype(int)

# Drop rows with NaN (from pct_change and shift)
df = df.dropna()

features = ['return_1h', 'return_3h', 'vol_change']
X = df[features]
y = df['target']

# Split: first 70% for training, last 30% for testing (no shuffling — time matters)
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.3)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
preds = model.predict(X_test)
accuracy = accuracy_score(y_test, preds)
print(f"Hit rate on test data: {accuracy:.1%}")

# Save the model to disk so other scripts can use it
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
print("Model saved to model.pkl")
