# prepare_raw_data.py

import pandas as pd
import argparse
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

def prepare_and_filter_data(input_file: str, output_file: str):
    logging.info(f"Loading raw data from: {input_file}")
    df = pd.read_csv(input_file)

    if 'Gmt time' in df.columns:
        df.rename(columns={'Gmt time': 'Date'}, inplace=True)
        logging.info("Renamed 'Gmt time' column to 'Date'.")
    elif 'gmt time' in df.columns:
        df.rename(columns={'gmt time': 'Date'}, inplace=True)
        logging.info("Renamed 'gmt time' column to 'Date'.")
    else:
        logging.error("Could not find 'Gmt time' column. Please check your input file.")
        return

    try:
        df['Date'] = pd.to_datetime(df['Date'], format='%d.%m.%Y %H:%M:%S.%f')
        logging.info("Converted 'Date' column to datetime objects.")
    except Exception as e:
        logging.error(f"Failed to convert date format. Please check your data. Error: {e}")
        return

    latest_date = df['Date'].max()
    cutoff_date = latest_date - pd.DateOffset(years=5)
    
    df_filtered = df[df['Date'] >= cutoff_date].copy()
    
    logging.info(f"Original data from {df['Date'].min()} to {latest_date}.")
    logging.info(f"Filtered data from {df_filtered['Date'].min()} to {df_filtered['Date'].max()} (Last 5 years).")
    logging.info(f"Kept {len(df_filtered):,} rows out of {len(df):,}.")

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_filtered.to_csv(output_path, index=False)
    logging.info(f"Successfully saved prepared data to: {output_path.resolve()}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Prepare and filter raw forex data.")
    parser.add_argument('--input-file', type=str, default="full_raw_data.csv", help="Path to the raw 20-year data CSV file.")
    parser.add_argument('--output-file', type=str, default="data_last_5_years.csv", help="Path to save the cleaned and filtered output CSV file.")
    args = parser.parse_args()
    
    prepare_and_filter_data(args.input_file, args.output_file)