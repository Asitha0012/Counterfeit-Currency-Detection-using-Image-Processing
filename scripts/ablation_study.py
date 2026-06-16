import os
import glob
import sys
import numpy as np

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(base_dir, 'src'))

from evaluate_lkr import analyze_lkr_note

def run_ablation():
    augmented_base = os.path.join(base_dir, "data", "augmented_testing")
    
    # We will test LKR 1000 Genuine and Fake
    gen_dir = os.path.join(augmented_base, "Genuine", "LKR_1000")
    fake_dir = os.path.join(augmented_base, "Fake", "LKR_1000")
    
    gen_imgs = glob.glob(os.path.join(gen_dir, '*.jpg'))[:50]
    fake_imgs = glob.glob(os.path.join(fake_dir, '*.jpg'))[:50]
    
    # Flat Voting (Baseline)
    flat_tp, flat_tn, flat_fp, flat_fn = 0, 0, 0, 0
    
    # Hybrid Architecture (Proposed)
    hybrid_tp, hybrid_tn, hybrid_fp, hybrid_fn = 0, 0, 0, 0
    
    print(f"Running Ablation Study on {len(gen_imgs)} Genuine and {len(fake_imgs)} Fake notes...")
    
    for img in gen_imgs:
        flat_verdict, robust_verdict, _, _, _, _ = analyze_lkr_note(img, "LKR_1000")
        if flat_verdict: flat_tp += 1
        else: flat_fn += 1
            
        if robust_verdict: hybrid_tp += 1
        else: hybrid_fn += 1
            
    for img in fake_imgs:
        flat_verdict, robust_verdict, _, _, _, _ = analyze_lkr_note(img, "LKR_1000")
        if not flat_verdict: flat_tn += 1
        else: flat_fp += 1
            
        if not robust_verdict: hybrid_tn += 1
        else: hybrid_fp += 1
        
    flat_acc = (flat_tp + flat_tn) / (len(gen_imgs) + len(fake_imgs))
    hybrid_acc = (hybrid_tp + hybrid_tn) / (len(gen_imgs) + len(fake_imgs))
    
    print("\n" + "="*50)
    print("ABLATION STUDY RESULTS (Accuracy)")
    print("="*50)
    print(f"Configuration 1: 75% Flat Voting Only")
    print(f"  Accuracy: {flat_acc*100:.1f}%")
    print(f"  False Positives (Fake passed as Gen): {flat_fp}")
    print(f"  False Negatives (Gen failed as Fake): {flat_fn}")
    print("\nConfiguration 2: Two-Stage Hybrid Architecture (Proposed)")
    print(f"  Accuracy: {hybrid_acc*100:.1f}%")
    print(f"  False Positives (Fake passed as Gen): {hybrid_fp}")
    print(f"  False Negatives (Gen failed as Fake): {hybrid_fn}")
    print("="*50)
    
if __name__ == "__main__":
    run_ablation()
