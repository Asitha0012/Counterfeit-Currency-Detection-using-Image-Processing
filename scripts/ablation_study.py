import os
import glob
import sys

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(base_dir, 'src'))

from evaluate_lkr import analyze_lkr_note
from evaluate_matrices import calculate_metrics

def test_architecture(image_paths, denom, use_veto_gates):
    y_true = []
    y_pred = []

    for img_path in image_paths:
        flat_verdict, robust_verdict, message, score, _, _ = analyze_lkr_note(img_path, denom)
        
        y_true.append("Fake") 
        
        if use_veto_gates:
            y_pred.append("Genuine" if robust_verdict else "Fake")
        else:
            y_pred.append("Genuine" if flat_verdict else "Fake")
            
    return y_true, y_pred

if __name__ == "__main__":
    augmented_base = os.path.join(base_dir, "data", "augmented_testing", "Fake")
    
    fake_1000 = glob.glob(os.path.join(augmented_base, "LKR_1000", '*.jpg'))
    fake_5000 = glob.glob(os.path.join(augmented_base, "LKR_5000", '*.jpg'))
    
    if not fake_1000 and not fake_5000:
        print("No fake images found!")
        sys.exit()
        
    print("\n==================================================")
    print("ABLATION STUDY: VETO GATES VS FLAT VOTING")
    print("==================================================")
    print("Testing the impact of Hybrid Veto Architecture on Synthetic Fakes")
    
    y_true_flat_1000, y_pred_flat_1000 = test_architecture(fake_1000, "LKR_1000", use_veto_gates=False)
    y_true_flat_5000, y_pred_flat_5000 = test_architecture(fake_5000, "LKR_5000", use_veto_gates=False)
    
    y_true_veto_1000, y_pred_veto_1000 = test_architecture(fake_1000, "LKR_1000", use_veto_gates=True)
    y_true_veto_5000, y_pred_veto_5000 = test_architecture(fake_5000, "LKR_5000", use_veto_gates=True)
    
    TP_flat, TN_f, FP_f, FN_flat, _, _, f1_flat, acc_flat = calculate_metrics(y_true_flat_1000 + y_true_flat_5000, y_pred_flat_1000 + y_pred_flat_5000)
    TP_veto, TN_v, FP_v, FN_veto, _, _, f1_veto, acc_veto = calculate_metrics(y_true_veto_1000 + y_true_veto_5000, y_pred_veto_1000 + y_pred_veto_5000)
    
    total_fakes = len(fake_1000) + len(fake_5000)
    
    print(f"\n[ARCHITECTURE A]: 75% Flat Voting Only (Baseline)")
    print(f"-> Caught Fakes: {TP_flat}/{total_fakes}")
    print(f"-> Fakes Escaped: {FN_flat}")
    print(f"-> Fake Detection Accuracy: {acc_flat*100:.1f}%\n")
    
    print(f"[ARCHITECTURE B]: Hybrid Veto Framework (Proposed)")
    print(f"-> Caught Fakes: {TP_veto}/{total_fakes}")
    print(f"-> Fakes Escaped: {FN_veto}")
    print(f"-> Fake Detection Accuracy: {acc_veto*100:.1f}%\n")
    
    print(f"SECURITY INCREASE: +{(acc_veto - acc_flat)*100:.1f}%")
