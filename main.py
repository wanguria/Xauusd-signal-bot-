import os
import sys
import time
import json
import hashlib
import logging
from datetime import datetime, timedelta
from pathlib import Path
import requests
import pandas as pd
import numpy as np

try:
    import ta
    from composio import Composio
except ImportError as e:
    print(f"Missing package: {e}. Run: pip install ta composio-core pandas requests")
    sys.exit(1)

# === CONFIG ===
OANDA_API_KEY = os.getenv("OANDA_API_KEY")
OANDA_ACCOUNT_ID = os.getenv("OANDA_ACCOUNT_ID")
OANDA_INSTRUMENT = "XAU_USD"
TIMEFRAME = "M5" # M5 = 5min candles
LOOKBACK_CANDLES = 500
MIN_STARS = 5 # Only 5-star signals = "high probability"
SESSION_HOURS_UTC = [13, 14, 15, 16] # London/NY overlap only
COMPOSIO_API_KEY = os.getenv("COMPOSIO_API_KEY")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
ALERT_EMAIL = os.getenv("ALERT_EMAIL")
DEDUP_FILE = Path("/tmp/xauusd_sent_signals.json")
LOG_LEVEL = "INFO"

# === LOGGING ===
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
log = logging.getLogger("XAUUSD-BOT")

# === COMPOSIO ===
composio = Composio(api_key=COMPOSIO_API_KEY)
entity = composio.get_entity(id="default")

def env_check():
    required = [OANDA_API_KEY, OANDA_ACCOUNT_ID, COMPOSIO_API_KEY, TELEGRAM_CHAT_ID, ALERT_EMAIL]
    if not all(required):
        log.error("Missing env vars. Need: OANDA_API_KEY, OANDA_ACCOUNT_ID, COMPOSIO_API_KEY, TELEGRAM_CHAT_ID, ALERT_EMAIL")
        sys.exit(1)

def get_oanda_data():
    url = f"https://api-fxpractice.oanda.com/v3/instruments/{OANDA_INSTRUMENT}/candles"
    headers = {"Authorization": f"Bearer {OANDA_API_KEY}"}
    params = {
        "count": LOOKBACK_CANDLES,
        "granularity": TIMEFRAME,
        "price": "M" # Mid prices
    }
    try:
        r = requests.get(url, headers=headers, params=params, timeout=10)
        r.raise_for_status()
        candles = r.json()['candles']
        df = pd.DataFrame([{
            'time': c['time'],
            'Open': float(c['mid']['o']),
            'High': float(c['mid']['h']),
            'Low': float(c['mid']['l']),
            'Close': float(c['mid']['c']),
            'Volume': int(c['volume'])
        } for c in candles if c['complete']])
        df['time'] = pd.to_datetime(df['time'])
        df.set_index('time', inplace=True)
        return df
    except Exception as e:
        log.error(f"OANDA fetch failed: {e}")
        raise

def detect_zones(df, consecutive=3, vol_mult=1.0):
    df['atr'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'], 14)
    df['vol_ma'] = df['Volume'].rolling(20).mean()
    supply, demand = [], []

    for i in range(50, len(df)-1):
        # Demand: 3 bullish candles + volume
        if all(df['Close'].iloc[i-j] > df['Open'].iloc[i-j] for j in range(consecutive)) and \
           df['Volume'].iloc[i] > df['vol_ma'].iloc[i] * vol_mult:
            zone = {
                'low': df['Low'].iloc[i-consecutive+1:i+1].min(),
                'high': df['High'].iloc[i-consecutive+1:i+1].max(),
                'time': df.index[i]
            }
            demand.append(zone)

        # Supply: 3 bearish candles + volume
        if all(df['Close'].iloc[i-j] < df['Open'].iloc[i-j] for j in range(consecutive)) and \
           df['Volume'].iloc[i] > df['vol_ma'].iloc[i] * vol_mult:
            zone = {
                'low': df['Low'].iloc[i-consecutive+1:i+1].min(),
                'high': df['High'].iloc[i-consecutive+1:i+1].max(),
                'time': df.index[i]
            }
            supply.append(zone)
    return supply, demand

def near_zone(price, zones, atr, max_atr=0.5):
    for z in zones[-8:]: # Check last 8 zones only
        if z['low'] - max_atr*atr <= price <= z['high'] + max_atr*atr:
            return True, z
    return False, None

def generate_signal(df):
    # Indicators
    df['sma200'] = ta.trend.sma_indicator(df['Close'], 200)
    df['sma9'] = ta.trend.sma_indicator(df['Close'], 9)
    df['sma20'] = ta.trend.sma_indicator(df['Close'], 20)
    df['macd'] = ta.trend.macd_diff(df['Close'], 12, 26, 9)
    df['adx'] = ta.trend.adx(df['High'], df['Low'], df['Close'], 14)
    df['vwap'] = ta.volume_weighted_average_price(df['High'], df['Low'], df['Close'], df['Volume'])
    df['rsi'] = ta.momentum.rsi(df['Close'], 14)
    df['vol_ma'] = df['Volume'].rolling(20).mean()
    df['atr'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'], 14)

    last, prev = df.iloc[-1], df.iloc[-2]
    supply_z, demand_z = detect_zones(df)

    near_demand, d_zone = near_zone(last['Close'], demand_z, last['atr'])
    near_supply, s_zone = near_zone(last['Close'], supply_z, last['atr'])

    # Base conditions
    long_base = prev['sma9'] < prev['sma20'] and last['sma9'] > last['sma20'] and last['macd'] > 0 and last['Close'] > last['sma200']
    short_base = prev['sma9'] > prev['sma20'] and last['sma9'] < last['sma20'] and last['macd'] < 0 and last['Close'] < last['sma200']

    if long_base and near_demand:
        stars = 2 # Base + Zone
        if last['rsi'] > 50: stars += 1
        if last['Volume'] > last['vol_ma']: stars += 1
        if last['adx'] > 25: stars += 1
        if last['Close'] > last['vwap']: stars += 1

        if stars >= MIN_STARS:
            sl = d_zone['low'] - last['atr'] * 0.2
            tp = last['Close'] + (last['Close'] - sl) * 2
            return "BUY", stars, last['Close'], sl, tp, "Demand Zone", df.index[-1]

    if short_base and near_supply:
        stars = 2
        if last['rsi'] < 50: stars += 1
        if last['Volume'] > last['vol_ma']: stars += 1
        if last['adx'] > 25: stars += 1
        if last['Close'] < last['vwap']: stars += 1

        if stars >= MIN_STARS:
            sl = s_zone['high'] + last['atr'] * 0.2
            tp = last['Close'] - (sl - last['Close']) * 2
            return "SELL", stars, last['Close'], sl, tp, "Supply Zone", df.index[-1]

    return None, 0, None, None, None, None, None

def is_duplicate(sig_hash):
    if not DEDUP_FILE.exists():
        return False
    data = json.loads(DEDUP_FILE.read_text())
    # Clear signals older than 4 hours
    cutoff = (datetime.utcnow() - timedelta(hours=4)).isoformat()
    data = {k:v for k,v in data.items() if v > cutoff}
    DEDUP_FILE.write_text(json.dumps(data))
    return sig_hash in data

def mark_sent(sig_hash):
    data = {}
    if DEDUP_FILE.exists():
        data = json.loads(DEDUP_FILE.read_text())
    data[sig_hash] = datetime.utcnow().isoformat()
    DEDUP_FILE.write_text(json.dumps(data))

def send_alert(signal, stars, price, sl, tp, setup, candle_time):
    if not signal: return

    # Session filter
    if datetime.utcnow().hour not in SESSION_HOURS_UTC:
        log.info(f"Outside session hours. Skip. Current UTC: {datetime.utcnow().hour}")
        return

    # Dedupe: hash = direction + candle_time
    sig_hash = hashlib.md5(f"{signal}{candle_time}".encode()).hexdigest()
    if is_duplicate(sig_hash):
        log.info("Duplicate signal. Already sent.")
        return

    msg = f"""🚨 XAUUSD {signal} {stars}⭐
Setup: {setup} + EMA9/20 + ADX + VWAP
Entry: ${price:.2f}
SL: ${sl:.2f} | TP: ${tp:.2f}
RR: 1:2 | Candle: {candle_time.strftime('%H:%M UTC')}
"""

    try:
        entity.execute(action="TELEGRAM_SEND_MESSAGE", params={"chat_id": TELEGRAM_CHAT_ID, "text": msg})
        entity.execute(action="GMAIL_SEND_EMAIL", params={
            "recipient_email": ALERT_EMAIL,
            "subject": f"XAUUSD {signal} {stars}⭐",
            "body": msg
        })
        mark_sent(sig_hash)
        log.info(f"Alert sent: {signal} @ {price:.2f}")
    except Exception as e:
        log.error(f"Composio send failed: {e}")

def run():
    env_check()
    log.info("Starting XAUUSD signal check...")
    try:
        df = get_oanda_data()
        log.info(f"Fetched {len(df)} candles. Last: {df.index[-1]} @ {df['Close'].iloc[-1]:.2f}")
        result = generate_signal(df)
        send_alert(*result)
    except Exception as e:
        log.error(f"Run failed: {e}")
        sys.exit(1)
    log.info("Check complete.")

if __name__ == "__main__":
    run()
