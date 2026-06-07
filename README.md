# Autonomous AI Trading POC

A minimal proof-of-concept for an AI-driven crypto trading agent.

## Architecture

Exchange API → TimescaleDB → ML Model → Trading Signal → Redis event bus

## Setup

### Prerequisites
- macOS with Homebrew
- Docker Desktop running
- Python 3.9+

### 1. Start infrastructure
```bash
docker run -d --name timescaledb -p 5432:5432 -e POSTGRES_PASSWORD=password timescale/timescaledb:latest-pg14
docker run -d --name redis -p 6379:6379 redis:latest
```

### 2. Create the database table
```bash
docker exec -it timescaledb psql -U postgres -d market -c "
CREATE TABLE IF NOT EXISTS ohlcv (
  ts timestamptz NOT NULL, symbol text NOT NULL,
  open float, high float, low float, close float, volume float,
  PRIMARY KEY (ts, symbol)
);
SELECT create_hypertable('ohlcv', 'ts', if_not_exists => TRUE);"
```

### 3. Install dependencies
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 4. Run the full pipeline
```bash
python orchestrator.py
```

## Components
- `fetch_data.py` — fetches OHLCV data from Binance via CCXT
- `train_model.py` — trains a RandomForest classifier
- `backtest.py` — evaluates strategy with Sharpe & Sortino ratios
- `orchestrator.py` — wires everything together via Redis pub/sub

## License
MIT
