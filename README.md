# tennis-betting-agent

# tennis-betting-ml-bot/

├── data/
│   ├── raw/                    # Raw data files (CSV/JSON)
│   └── processed/              # Cleaned feature sets
├── ingest.py                  # Collects live/historical data
├── features.py                # Cleans data, does feature engineering
├── train.py                   # ML training pipeline
├── predict.py                 # Run inference and send alerts
├── telegram_config.py         # Stores Telegram bot API key, chat ID
├── telegram_alert.py          # Sends predictions via Telegram
├── model.pkl                  # Trained model artifact
├── requirements.txt           # Dependencies
├── Dockerfile                 # For deployment (optional)
├── .github/
│   └── workflows/
│       └── pipeline.yml       # CI/CD automation using GitHub Actions
├── README.md                  # Project overview

# ================= ingest.py =================
import requests
import pandas as pd
import os

def fetch_historical():
    url = "https://tennis-data.co.uk/2024/atp_matches.csv"
    df = pd.read_csv(url)
    df.to_csv("data/raw/historical.csv", index=False)

def fetch_live():
    # Placeholder: Replace with real API
    print("Fetching live data...")

if __name__ == '__main__':
    fetch_historical()
    fetch_live()

# ================= features.py =================
import pandas as pd
import numpy as np

RAW = "data/raw/historical.csv"
PROCESSED = "data/processed/features.csv"

def create_features():
    df = pd.read_csv(RAW)
    df = df.dropna()
    df['elo_diff'] = df['player1_elo'] - df['player2_elo']
    df['surface_encoded'] = df['surface'].map({'Hard': 0, 'Clay': 1, 'Grass': 2})
    df['target'] = (df['winner'] == df['player1']).astype(int)
    df.to_csv(PROCESSED, index=False)

if __name__ == '__main__':
    create_features()

# ================= train.py =================
import pandas as pd
import xgboost as xgb
import joblib

PROCESSED = "data/processed/features.csv"

if __name__ == '__main__':
    df = pd.read_csv(PROCESSED)
    X = df[['elo_diff', 'surface_encoded']]
    y = df['target']
    model = xgb.XGBClassifier()
    model.fit(X, y)
    joblib.dump(model, "model.pkl")

# ================= predict.py =================
import joblib
import pandas as pd
from telegram_alert import send_alert

model = joblib.load("model.pkl")

# Mock: Live match features
live = pd.DataFrame({
    'elo_diff': [42],
    'surface_encoded': [0],
})

prob = model.predict_proba(live)[0][1]
if prob > 0.6:
    send_alert(f"High confidence bet: P1 win prob = {prob:.2f}")

# ================= telegram_alert.py =================
import requests
from telegram_config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID

def send_alert(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    requests.post(url, data={"chat_id": TELEGRAM_CHAT_ID, "text": message})

# ================= telegram_config.py =================
TELEGRAM_BOT_TOKEN = "your-telegram-bot-token"
TELEGRAM_CHAT_ID = "your-chat-id"

# ================= requirements.txt =================
pandas
xgboost
joblib
requests

# ================= .github/workflows/pipeline.yml =================
name: Tennis ML Bot Pipeline

on:
  push:
  schedule:
    - cron: '0 4 * * *'

jobs:
  run-pipeline:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Setup Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.10'
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Ingest Data
        run: python ingest.py
      - name: Create Features
        run: python features.py
      - name: Train Model
        run: python train.py
      - name: Predict + Notify
        run: python predict.py
