import glob
import os
import sys
import time

sys.path.append('src')
from evaluate_lkr import analyze_lkr_note

def evaluate_all():
    print("="*60)
    print("FAKE CURRENCY DETECTION - FULL PROJECT EVALUATION")
    print("="*60)
    
    datasets = [
        ('Genuine LKR 1000', 'data/genuine/LKR_1000', 'LKR_1000', True),
        ('Genuine LKR 5000', 'data/genuine/LKR_5000', 'LKR_5000', True),
        ('Counterfeit LKR 1000', 'data/counterfeit/LKR_1000', 'LKR_1000', False),
        ('Counterfeit LKR 5000', 'data/counterfeit/LKR_5000', 'LKR_5000', False),
    ]
    
    total_notes = 0
    total_correct = 0
    
    start_time = time.time()
    
    for name, glob_dir, denom_str, is_genuine in datasets:
        files = []
        for ext in ('*.jpg', '*.jpeg', '*.png'):
            files.extend(glob.glob(os.path.join(glob_dir, ext)))
        
        if not files:
            continue
            
        print(f"\nEvaluating {name} ({len(files)} notes)...")
        
        correct = 0
        for f in files:
            flat_verdict, robust_verdict, message, score, _, _ = analyze_lkr_note(f, denom_str)
            
            # True means Genuine, False means Fake
            if robust_verdict == is_genuine:
                correct += 1
                status = "PASS"
            else:
                status = "FAIL"
                
            print(f"  [{status}] {os.path.basename(f)} - Score: {score}/12 -> Robust Verdict: {'GENUINE' if robust_verdict else 'FAKE'}")
            
        print(f"  -> Accuracy for {name}: {correct}/{len(files)} ({(correct/len(files))*100:.1f}%)")
        total_notes += len(files)
        total_correct += correct
        
    print("\n" + "="*60)
    print(f"FINAL OVERALL ACCURACY: {total_correct}/{total_notes} ({(total_correct/total_notes)*100:.1f}%)")
    print(f"Evaluation completed in {time.time() - start_time:.2f} seconds.")
    print("="*60)

if __name__ == '__main__':
    evaluate_all()
