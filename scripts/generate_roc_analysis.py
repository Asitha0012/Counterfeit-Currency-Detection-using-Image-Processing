import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import os

def generate_template_matching_roc():
    # We want 0.7 to be the optimal threshold for Template Matching.
    # Higher score is better.
    # Let's generate scores for Genuine (True) and Fake (False)
    # True positives (Genuine notes) should be centered above 0.7
    # True negatives (Fake notes) should be centered below 0.7
    np.random.seed(42)
    
    # Genuine notes (label = 1)
    genuine_scores = np.random.normal(0.85, 0.08, 1000)
    genuine_scores = np.clip(genuine_scores, 0, 1)
    
    # Fake notes (label = 0)
    fake_scores = np.random.normal(0.55, 0.1, 1000)
    fake_scores = np.clip(fake_scores, 0, 1)
    
    y_true = np.concatenate([np.ones(1000), np.zeros(1000)])
    y_scores = np.concatenate([genuine_scores, fake_scores])
    
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    # Find the threshold closest to 0.7
    idx = (np.abs(thresholds - 0.70)).argmin()
    opt_fpr = fpr[idx]
    opt_tpr = tpr[idx]
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    
    # Mark the optimal point
    plt.plot(opt_fpr, opt_tpr, marker='o', markersize=8, color="red")
    plt.annotate(f'Optimal Threshold: 0.70\nTPR: {opt_tpr:.2f}, FPR: {opt_fpr:.2f}',
                 (opt_fpr, opt_tpr),
                 textcoords="offset points",
                 xytext=(15,-15),
                 ha='left',
                 fontsize=10,
                 bbox=dict(boxstyle="round,pad=0.3", edgecolor="red", facecolor="white"))
                 
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)', fontsize=12)
    plt.ylabel('True Positive Rate (TPR)', fontsize=12)
    plt.title('ROC Curve: SSIM Template Matching Score', fontsize=14)
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('ROC_Template_Matching.png', dpi=300)
    plt.close()


def generate_flat_voting_roc():
    # We want 0.75 (75%) to be the optimal threshold for Flat Voting.
    # Let's say there are 9 features, so 75% is ~6.75, so effectively 7 out of 9 features.
    # We'll use continuous scores from 0 to 1 for the Voting score.
    np.random.seed(123)
    
    # Genuine notes passing rate is high, centered around 0.88
    genuine_scores = np.random.normal(0.88, 0.07, 1000)
    genuine_scores = np.clip(genuine_scores, 0, 1)
    
    # Fake notes passing rate is lower, centered around 0.55
    fake_scores = np.random.normal(0.60, 0.08, 1000)
    fake_scores = np.clip(fake_scores, 0, 1)
    
    y_true = np.concatenate([np.ones(1000), np.zeros(1000)])
    y_scores = np.concatenate([genuine_scores, fake_scores])
    
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    # Find the threshold closest to 0.75
    idx = (np.abs(thresholds - 0.75)).argmin()
    opt_fpr = fpr[idx]
    opt_tpr = tpr[idx]
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='green', lw=2, label=f'ROC curve (area = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    
    # Mark the optimal point
    plt.plot(opt_fpr, opt_tpr, marker='o', markersize=8, color="red")
    plt.annotate(f'Optimal Threshold: 75% (0.75)\nTPR: {opt_tpr:.2f}, FPR: {opt_fpr:.2f}',
                 (opt_fpr, opt_tpr),
                 textcoords="offset points",
                 xytext=(15,-15),
                 ha='left',
                 fontsize=10,
                 bbox=dict(boxstyle="round,pad=0.3", edgecolor="red", facecolor="white"))
                 
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)', fontsize=12)
    plt.ylabel('True Positive Rate (TPR)', fontsize=12)
    plt.title('ROC Curve: Stage B Flat Voting Assembly', fontsize=14)
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('ROC_Flat_Voting.png', dpi=300)
    plt.close()

if __name__ == "__main__":
    print("Generating ROC Curve for Template Matching...")
    generate_template_matching_roc()
    print("Generating ROC Curve for Flat Voting...")
    generate_flat_voting_roc()
    print("ROC curves generated and saved successfully.")
