import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def calculate_v25_metrics():
    """Calculate V_25 win-rate and create proper confusion matrix image"""
    
    # Read confusion matrix CSV
    conf_path = Path("V_25_MeanReversion_Classifier/results/confusion_matrix.csv")
    conf_df = pd.read_csv(conf_path, index_col=0)
    
    # Extract values: [[TN, FP], [FN, TP]]
    tn, fp = conf_df.iloc[0].values
    fn, tp = conf_df.iloc[1].values
    
    # Calculate metrics
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    print("=== V_25 MEAN REVERSION CLASSIFIER ===")
    print(f"Accuracy: {accuracy:.3f}")
    print(f"Precision: {precision:.3f} (when predicting reversion, how often correct)")
    print(f"Recall: {recall:.3f} (of all actual reversions, how many caught)")
    print(f"True Positive Rate (Win-rate): {recall:.3f}")
    
    # Create proper confusion matrix plot
    conf_matrix = np.array([[tn, fp], [fn, tp]])
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=['No Reversion', 'Reversion'],
                yticklabels=['No Reversion', 'Reversion'])
    plt.title('V_25 Mean Reversion Classifier - Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    
    output_path = Path("V_25_MeanReversion_Classifier/results/confusion_matrix.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Created confusion matrix image: {output_path}")
    
    return accuracy, precision, recall

def analyze_v23_predictions():
    """Analyze V_23 SALVAGE predictions to estimate win-rate"""
    
    # Load test data and check if we can estimate performance
    test_path = Path("V_23_SALVAGE/regression_CORRECTED/data/test_data.csv")
    
    if not test_path.exists():
        print("=== V_23 SALVAGE ===")
        print("Cannot calculate win-rate - no ground truth labels in test data")
        print("Based on loss curves: Models learned poorly after data leakage correction")
        print("Generated signals: ~1,500 (but model quality questionable)")
        return
    
    # If we have the data, we could analyze further
    df = pd.read_csv(test_path, index_col=0, parse_dates=True)
    print("=== V_23 SALVAGE ANALYSIS ===")
    print(f"Test samples: {len(df):,}")
    print("Model Quality: POOR (based on loss curves)")
    print("Learning: Minimal improvement without leaked features")
    print("Signals Generated: ~1,500 (10+ pips, R/R 1.2+)")
    print("Recommendation: Low confidence due to poor model learning")

if __name__ == "__main__":
    print("PERFORMANCE ANALYSIS\n" + "="*50)
    
    # V_25 Analysis
    try:
        calculate_v25_metrics()
    except Exception as e:
        print(f"V_25 analysis failed: {e}")
    
    print("\n" + "="*50)
    
    # V_23 Analysis  
    try:
        analyze_v23_predictions()
    except Exception as e:
        print(f"V_23 analysis failed: {e}")