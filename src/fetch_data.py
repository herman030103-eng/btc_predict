import requests
import time
import argparse
import os
import logging
import pandas as pd
import datetime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("fetch_data")

BASE_URL = "https://api.binance.com/api/v3/klines"

def fetch_klines(symbol, interval, limit=1000, start_time=None):
    params = {
        "symbol": symbol,
        "interval": interval,
        "limit": limit,
    }
    if start_time is not None:
        params["startTime"] = start_time

    response = requests.get(BASE_URL, params=params)
    if response.status_code != 200:
        logger.error(f"Error fetching data: {response.status_code} {response.text}")
        return []

    data = response.json()
    return data

def fetch_all_klines(symbol, interval, start_time=None):
    limit_per_request = 1000
    all_klines = []

    current_start_time = start_time
    while True:
        klines = fetch_klines(symbol, interval, limit=limit_per_request, start_time=current_start_time)
        if not klines:
            logger.info("No more data returned by API, finishing.")
            break

        all_klines.extend(klines)
        last_close_time = klines[-1][6]  # close_time индекс 6
        current_start_time = last_close_time + 1

        last_close_dt = datetime.datetime.fromtimestamp(last_close_time / 1000)
        logger.info(f"Fetched {len(all_klines)} klines. Last close time: {last_close_dt}")

        # Если получили меньше, чем limit_per_request — значит данных больше нет
        if len(klines) < limit_per_request:
            logger.info("Reached end of available data.")
            break

        time.sleep(0.1)

    return all_klines

def klines_to_dataframe(klines):
    columns = [
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "number_of_trades",
        "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"
    ]
    df = pd.DataFrame(klines, columns=columns)
    # Приводим типы
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms")
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="BTCUSDT", help="Trading pair symbol")
    parser.add_argument("--interval", default="1h", help="Kline interval")
    parser.add_argument("--start_date", default=None, help="Start date YYYY-MM-DD to begin fetching data")
    parser.add_argument("--output", default=None, help="Output CSV filename")
    args = parser.parse_args()

    if not args.start_date:
        logger.error("Please specify --start_date (e.g. 2023-01-01)")
        return

    output_file = args.output or f"data/raw_{args.symbol}_{args.interval}.csv"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    dt = datetime.datetime.strptime(args.start_date, "%Y-%m-%d")
    start_time = int(dt.timestamp() * 1000)  # в миллисекундах

    logger.info(f"Starting fetching klines for {args.symbol} at interval {args.interval} from {args.start_date} until now")

    klines = fetch_all_klines(args.symbol, args.interval, start_time=start_time)
    if not klines:
        logger.error("No klines fetched. Exiting.")
        return

    df = klines_to_dataframe(klines)
    df.to_csv(output_file, index=False)
    logger.info(f"Saved klines to {output_file} — rows: {len(df)}")

if __name__ == "__main__":
    main()
