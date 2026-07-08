import sys
import os
import glob

sys.path.append('src')
from evaluate_lkr import analyze_lkr_note

def main():
    denoms = ['LKR_500', 'LKR_1000', 'LKR_5000']
    for denom in denoms:
        print(f"\n========================================")
        print(f"EVALUATING {denom} GENUINE NOTES")
        print(f"========================================")
        genuine_dir = os.path.join('data', 'genuine', denom)
        img_paths = sorted(glob.glob(os.path.join(genuine_dir, '*')))
        
        for path in img_paths:
            print(f"\n--- {os.path.basename(path)} ---")
            flat_verdict, robust_verdict, message, score, feature_statuses, _ = analyze_lkr_note(path, denom)
            for i, (passed, msg) in enumerate(feature_statuses):
                print(f"Feature {i+1}: {msg}")

if __name__ == '__main__':
    main()
