import cv2
import sys
import os

sys.path.append('src')
from align_note import align_note
from features.programmatic_features import verify_security_thread

def test_f11_widths():
    print(f"{'Note':<10} | {'Passed':<6} | {'Message':<50}")
    print("-" * 70)
    
    for note_type in ['G', 'F']:
        for i in range(1, 11):
            dir_name = 'genuine' if note_type == 'G' else 'counterfeit'
            img_path = f"data/{dir_name}/LKR_500/500_{note_type}{i}.jpg"
            if not os.path.exists(img_path): continue
            
            img = cv2.imread(img_path)
            aligned = align_note(img, 'LKR_500')
            
            passed, msg, _ = verify_security_thread(aligned, 'LKR_500')
            print(f"{f'500_{note_type}{i}':<10} | {str(passed):<6} | {msg:<50}")

test_f11_widths()
