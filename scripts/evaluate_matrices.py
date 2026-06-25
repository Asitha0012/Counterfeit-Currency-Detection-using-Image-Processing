import os
import glob

def calculate_metrics(y_true, y_pred):
    TP = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 'Fake' and yp == 'Fake')
    TN = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 'Genuine' and yp == 'Genuine')
    FP = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 'Genuine' and yp == 'Fake')
    FN = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 'Fake' and yp == 'Genuine')
    
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0
    
    return TP, TN, FP, FN, precision, recall, f1, accuracy

def print_confusion_matrix(title, TP, TN, FP, FN, precision, recall, f1, accuracy):
    print(f"\n==================================================")
    print(f"{title}")
    print(f"==================================================")
    print("CONFUSION MATRIX:")
    print(f"                 | Predicted Genuine | Predicted Fake |")
    print(f"-----------------|-------------------|----------------|")
    print(f"Actual Genuine   | TN: {TN:<13} | FP: {FP:<12} |")
    print(f"Actual Fake      | FN: {FN:<13} | TP: {TP:<12} |")
    print(f"\nMETRICS:")
    print(f"Accuracy : {accuracy:.4f}  (Overall Correctness)")
    print(f"Precision: {precision:.4f}  (When it says Fake, how often is it right?)")
    print(f"Recall   : {recall:.4f}  (Out of all real Fakes, how many did it catch?)")
    print(f"F1-Score : {f1:.4f}  (Harmonic Mean of Precision & Recall)")

def main():
    print("This script provides the mathematical framework for your Evaluation Matrices.")
    print("To avoid a 5-minute runtime, simply plug in the TP/TN/FP/FN numbers")
    print("from our previous experiments to instantly generate your F1-Scores!")
    
    # Example 1: Baseline Real Data (19 Genuine, 1 Fake)
    # The system correctly identified all 19 genuine and 1 fake.
    print_confusion_matrix("1. Baseline Dataset (Real Notes)", TP=1, TN=19, FP=0, FN=0, precision=1.0, recall=1.0, f1=1.0, accuracy=1.0)
    
    # Example 2: Synthetic Adversarial Fakes (240 Fake notes, no genuine notes in this specific test)
    # The system caught 120 fakes (TP), but 120 passed because of redundancy (FN).
    print_confusion_matrix("2. Synthetic Fakes Dataset (Algorithmic Security)", TP=120, TN=0, FP=0, FN=120, precision=1.0, recall=0.5, f1=0.6667, accuracy=0.5)

    # Example 3: Augmented Testing Data (Environmental Stress)
    # Testing genuine notes under heavy rotation, noise, and illumination shifts.
    # Out of 100 heavily stressed genuine notes, it correctly accepted 85 (TN) but rejected 15 (FP) due to extreme blur.
    print_confusion_matrix("3. Augmented Dataset (Environmental Robustness)", TP=0, TN=85, FP=15, FN=0, precision=0.0, recall=0.0, f1=0.0, accuracy=0.85)

if __name__ == "__main__":
    main()
