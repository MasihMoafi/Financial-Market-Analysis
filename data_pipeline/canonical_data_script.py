# generate_data_canonical.py
#
# --- Summary of Changes ---
# 1. CORRECT LABELING: Implements the correct logic where trades are entered at the
#    'Open' of the *next* candle, not the 'Close' of the signal candle.
# 2. NEAR-MISS ANALYSIS: Includes the experiment to test how often price nearly
#    hits the Take-Profit before reversing to hit the Stop-Loss.
# 3. CLEAN & FOCUSED: This script has one job: generate clean, valid data.
# -----------------------------------------

import pandas as pd
import numpy as np
import logging
import argparse
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

def generate_tpsl_signal_corrected(df: pd.DataFrame, lookahead_hours: int, tp_pips: int, sl_pips: int) -> pd.DataFrame:
    """
    Generates a 3-class signal ('buy', 'sell', 'keep') with two critical corrections:
    1. Entry price is the Open of the candle *after* the signal candle.
    2. Includes analysis for the 'near-miss TP' problem.
    """
    logging.info(f"Generating signals: {lookahead_hours}h lookahead, TP={tp_pips}, SL={sl_pips}")
    
    # --- Setup ---
    pip_multiplier = 0.0001
    tp_points = tp_pips * pip_multiplier
    sl_points = sl_pips * pip_multiplier
    lookahead_candles = lookahead_hours * 60 # Assuming 1-minute data

    # Get numpy arrays for performance
    high = df['high'].to_numpy()
    low = df['low'].to_numpy()
    # **CRITICAL CORRECTION**: Get the open of the *next* candle for entry price
    entry_price = df['open'].shift(-1).to_numpy()
    
    n = len(df)
    signal = np.full(n, 'keep', dtype=object)

    # --- Near-Miss Analysis Setup ---
    near_miss_counter = 0
    total_sl_hits = 0
    near_miss_threshold_factor = 0.95 # Price must get within 5% of TP to be a "near miss"

    # Loop must end early to account for lookahead window and next-candle entry
    for i in range(n - lookahead_candles - 1):
        entry = entry_price[i]
        
        # If entry price is invalid (e.g., NaN at the end), skip
        if np.isnan(entry):
            continue

        # Define the lookahead window starting from the candle *after* entry
        window = slice(i + 1, i + 1 + lookahead_candles)
        window_high = np.max(high[window])
        window_low = np.min(low[window])
        
        # --- Check for TP/SL hits based on the correct `entry` price ---
        buy_tp_hit = window_high >= entry + tp_points
        buy_sl_hit = window_low <= entry - sl_points
        
        sell_tp_hit = window_low <= entry - tp_points
        sell_sl_hit = window_high >= entry + sl_points

        # Find the time of the first hit
        buy_tp_time = np.argmax(high[window] >= entry + tp_points) if buy_tp_hit else np.inf
        buy_sl_time = np.argmax(low[window] <= entry - sl_points) if buy_sl_hit else np.inf

        sell_tp_time = np.argmax(low[window] <= entry - tp_points) if sell_tp_hit else np.inf
        sell_sl_time = np.argmax(high[window] >= entry + sl_points) if sell_sl_hit else np.inf

        # --- Assign Signal based on which was hit first ---
        if buy_tp_time < buy_sl_time:
            signal[i] = 'buy'
        elif sell_tp_time < sell_sl_time:
            signal[i] = 'sell'
        else:
            # --- NEAR-MISS ANALYSIS ---
            # This block runs only if the final label is 'keep' or a loss.
            # We check if a loss was preceded by a near-miss take profit.
            
            # Check for a "near-miss buy" (SL was hit, but price got close to TP first)
            if buy_sl_hit and not buy_tp_hit:
                total_sl_hits += 1
                if window_high >= entry + (tp_points * near_miss_threshold_factor):
                    near_miss_counter += 1
            
            # Check for a "near-miss sell"
            elif sell_sl_hit and not sell_tp_hit:
                total_sl_hits += 1
                if window_low <= entry - (tp_points * near_miss_threshold_factor):
                    near_miss_counter += 1

    # --- Log the Near-Miss Analysis Results ---
    if total_sl_hits > 0:
        near_miss_percentage = (near_miss_counter / total_sl_hits) * 100
        logging.info("--- Near-Miss Analysis Results ---")
        logging.info(f"Total Stop-Loss Hits Evaluated: {total_sl_hits}")
        logging.info(f"Trades that nearly hit TP (within 5%) before hitting SL: {near_miss_counter}")
        logging.info(f"This accounts for {near_miss_percentage:.2f}% of all losses.")
        logging.info("---------------------------------")
    else:
        logging.warning("No stop-loss hits were recorded, near-miss analysis cannot be performed.")

    df['signal'] = signal
    logging.info(f"Final 3-class signal distribution:\n{df['signal'].value_counts(normalize=True)}")
    
    # Remove the last rows that couldn't be labeled
    df.dropna(subset=['signal'], inplace=True)
    return df

def main(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"--- Starting Canonical Dataset Creation ---")
    
    # Load and prepare the data
    df = pd.read_csv(args.input_file, parse_dates=True)
    # Ensure standard column names
    df.columns = [x.lower() for x in df.columns]
    df.rename(columns={'date': 'Date', 'time': 'Date'}, inplace=True) # Standardize date column
    df = df.set_index('Date').sort_index()

    # Generate signals with the corrected logic
    df_final = generate_tpsl_signal_corrected(
        df, args.lookahead_hours, args.tp_pips, args.sl_pips
    )
    
    # Add stationary return features for the model
    prev_close = df_final['close'].shift(1)
    df_final['open_return'] = (df_final['open'] - prev_close) / prev_close
    df_final['high_return'] = (df_final['high'] - df_final['open']) / df_final['open']
    df_final['low_return'] = (df_final['low'] - df_final['open']) / df_final['open']
    df_final['close_return'] = (df_final['close'] - df_final['open']) / df_final['open']
    df_final['Body'] = abs(df_final['close'] - df_final['open'])
    df_final['High-Low'] = df_final['high'] - df_final['low']
    df_final.dropna(inplace=True)
    
    # Split into train and test sets
    split_index = int(len(df_final) * (1 - args.test_size))
    train_df = df_final.iloc[:split_index]
    test_df = df_final.iloc[split_index:]
    
    # Save the files
    train_path = output_dir / 'train.csv'
    test_path = output_dir / 'test.csv'
    train_df.to_csv(train_path)
    test_df.to_csv(test_path)
    
    logging.info(f"Train set: {len(train_df):,} rows | Test set: {len(test_df):,} rows")
    logging.info(f"Successfully saved files to {train_path.resolve()} and {test_path.resolve()}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate canonical datasets with corrected TP/SL logic and near-miss analysis.")
    parser.add_argument('--input-file', type=str, default='data_last_5_years.csv', help="Path to the raw input CSV file.")
    parser.add_argument('--output-dir', type=str, default="processed_data_canonical", help="Directory to save the train.csv and test.csv files.")
    parser.add_argument('--lookahead-hours', type=int, default=24, help="Lookahead period in hours.")
    parser.add_argument('--tp-pips', type=int, default=20, help="Take-Profit distance in pips.")
    parser.add_argument('--sl-pips', type=int, default=10, help="Stop-Loss distance in pips.")
    parser.add_argument('--test-size', type=float, default=0.2, help="Proportion of the data to use for the test set.")
    args = parser.parse_args()
    main(args)