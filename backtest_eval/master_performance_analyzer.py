

import pandas as pd
import numpy as np
import argparse
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

def calculate_financial_metrics(predictions_df: pd.DataFrame, tp_pips: int, sl_pips: int, lot_size: float = 0.1):
    """
    Calculates financial performance metrics from a predictions file.

    Args:
        predictions_df (pd.DataFrame): DataFrame with 'signal' and 'predicted_signal'.
        tp_pips (int): Take-profit in pips.
        sl_pips (int): Stop-loss in pips.
        lot_size (float): The lot size for each trade.

    Returns:
        dict: A dictionary containing performance metrics.
    """
    logging.info(f"Calculating metrics for {len(predictions_df)} predictions with TP={tp_pips}, SL={sl_pips}")

    # --- P&L Calculation ---
    # We determine the result of a trade based on the true 'signal' column.
    # The model's prediction determines IF a trade was taken.
    
    # 1 pip movement in EUR/USD for a 0.1 lot size is roughly $1.
    pip_value = lot_size * 10 

    profit = tp_pips * pip_value
    loss = -sl_pips * pip_value

    # Conditions for P&L
    conditions = [
        (predictions_df['predicted_signal'] == 'buy') & (predictions_df['signal'] == 'buy'),   # Successful Buy
        (predictions_df['predicted_signal'] == 'buy') & (predictions_df['signal'] == 'sell'),  # Failed Buy (hits SL)
        (predictions_df['predicted_signal'] == 'sell') & (predictions_df['signal'] == 'sell'), # Successful Sell
        (predictions_df['predicted_signal'] == 'sell') & (predictions_df['signal'] == 'buy'),  # Failed Sell (hits SL)
    ]
    outcomes = [profit, loss, profit, loss]
    
    predictions_df['pnl'] = np.select(conditions, outcomes, default=0)
    
    trades_taken = predictions_df[predictions_df['predicted_signal'] != 'keep']
    if len(trades_taken) == 0:
        logging.warning("No trades were taken according to the predictions. Cannot calculate metrics.")
        return None

    # --- Metric Calculation ---
    total_pnl = trades_taken['pnl'].sum()
    win_trades = trades_taken[trades_taken['pnl'] > 0]
    loss_trades = trades_taken[trades_taken['pnl'] < 0]
    
    win_rate = len(win_trades) / len(trades_taken) if len(trades_taken) > 0 else 0
    
    # Sharpe Ratio (simplified: assuming risk-free rate is 0)
    daily_returns = trades_taken['pnl'].resample('D').sum()
    sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252) if daily_returns.std() > 0 else 0

    metrics = {
        "Total Trades": len(trades_taken),
        "Winning Trades": len(win_trades),
        "Losing Trades": len(loss_trades),
        "Win Rate (%)": win_rate * 100,
        "Total P&L ($)": total_pnl,
        "Sharpe Ratio (Annualized)": sharpe_ratio,
        "Average P&L per Trade ($)": trades_taken['pnl'].mean()
    }
    
    return metrics

def main():
    parser = argparse.ArgumentParser(description="Master Performance Analyzer for Forex Models")
    parser.add_argument('--predictions-file', type=str, required=True, help="Path to the prediction CSV file. Must contain 'signal' and 'predicted_signal' columns.")
    parser.add_argument('--tp-pips', type=int, default=20, help="Take-Profit in pips for the calculation.")
    parser.add_argument('--sl-pips', type=int, default=10, help="Stop-Loss in pips for the calculation.")
    args = parser.parse_args()

    pred_path = Path(args.predictions_file)
    if not pred_path.exists():
        logging.error(f"Predictions file not found at: {pred_path}")
        return

    logging.info(f"Loading predictions from {pred_path}")
    df = pd.read_csv(pred_path, index_col='date', parse_dates=True) # Assuming 'date' is the index

    # Ensure required columns exist
    if 'signal' not in df.columns or 'predicted_signal' not in df.columns:
        logging.error("CSV must contain 'signal' and 'predicted_signal' columns.")
        return

    results = calculate_financial_metrics(df, args.tp_pips, args.sl_pips)

    if results:
        print("\n--- Financial Performance Report ---")
        for key, value in results.items():
            print(f"{key:<30}: {value:,.2f}")
        print("------------------------------------")

if __name__ == '__main__':
    main()

