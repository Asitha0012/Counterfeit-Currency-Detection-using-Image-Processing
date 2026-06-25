import os
import glob
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(base_dir, 'src'))

from evaluate_lkr import analyze_lkr_note

def collect_scores():
    augmented_base = os.path.join(base_dir, "data", "augmented_testing")
    
    genuine_imgs = glob.glob(os.path.join(augmented_base, "Genuine", "**", "*.jpg"), recursive=True)
    fake_imgs = glob.glob(os.path.join(augmented_base, "Fake", "**", "*.jpg"), recursive=True)
    
    y_true = []
    y_scores = []
    
    print(f"Extracting scores from {len(genuine_imgs)} Genuine notes...")
    for img_path in genuine_imgs:
        denom = "LKR_1000" if "LKR_1000" in img_path else "LKR_5000"
        _, _, _, score, _, _ = analyze_lkr_note(img_path, denom)
        y_true.append(1)
        y_scores.append(score / 100.0)
        
    print(f"Extracting scores from {len(fake_imgs)} Fake notes...")
    for img_path in fake_imgs:
        denom = "LKR_1000" if "LKR_1000" in img_path else "LKR_5000"
        _, _, _, score, _, _ = analyze_lkr_note(img_path, denom)
        y_true.append(0)
        y_scores.append(score / 100.0)
        
    return np.array(y_true), np.array(y_scores)

def plot_roc(y_true, y_scores):
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (Counterfeits passing as Genuine)')
    plt.ylabel('True Positive Rate (Genuine Notes Passed)')
    plt.title('Receiver Operating Characteristic (ROC) - Stage B Flat Voting')
    plt.legend(loc="lower right")
    
    # Save the figure to the artifacts directory so the user can see it!
    out_path = os.path.join(base_dir, 'ROC_Curve.png')
    plt.savefig(out_path)
    print(f"ROC curve saved to {out_path}")
    
if __name__ == "__main__":
    y_true, y_scores = collect_scores()
    plot_roc(y_true, y_scores)
    print("ROC Analysis Complete!")
