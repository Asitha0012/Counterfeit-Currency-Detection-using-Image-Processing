import sys
import os
import glob
sys.path.append('src')
from evaluate_lkr import analyze_lkr_note

def test_12_features():
    print("Evaluating all 10 LKR_5000 Genuine notes for 12/12 features:")
    genuine_dir = os.path.join('data', 'genuine', 'LKR_5000')
    img_paths = sorted(glob.glob(os.path.join(genuine_dir, '*')))
    
    for path in img_paths:
        flat_verdict, robust_verdict, message, score, feature_statuses, _ = analyze_lkr_note(path, 'LKR_5000')
        passed_features = sum(1 for f in feature_statuses if f[0])
        
        status = "PASS" if robust_verdict else "FAIL"
        print(f"{os.path.basename(path)} -> {status} ({passed_features}/12 features passed)")
        if not robust_verdict or passed_features < 12:
            failed = [i+1 for i, f in enumerate(feature_statuses) if not f[0]]
            print(f"   Failed features: {failed}")

if __name__ == '__main__':
    test_12_features()
