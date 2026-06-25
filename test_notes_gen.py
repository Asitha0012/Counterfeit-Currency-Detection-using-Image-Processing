import os
import cv2
import sys

sys.path.append('src')
from evaluate_lkr import analyze_lkr_note

def evaluate_note(path, denom):
    print(f"\n--- Evaluating {os.path.basename(path)} ---")
    flat, robust, msg, score, statuses, images = analyze_lkr_note(path, denom)
    
    feature_names = [
        "F1: Micro-Printing", "F2: Butterfly", "F3: Note Value", 
        "F4: Bird", "F5: Lion Emblem", "F6: Value Text", "F7: Watermark",
        "F8: Blind Dots", "F9: Asymmetric Serial", "F10: Vertical Red Serial", 
        "F11: Security Thread", "F12: Edge Lines"
    ]
    
    for i, (passed, fmsg) in enumerate(statuses):
        if i < len(feature_names):
            print(f"{feature_names[i]}: {'PASS' if passed else 'FAIL'} - {fmsg}")

notes = [
    (r'data\genuine\LKR_1000\1000_G1.jpg', 'LKR_1000'),
    (r'data\genuine\LKR_1000\1000_G3.jpg', 'LKR_1000'),
    (r'data\genuine\LKR_1000\1000_G6.jpg', 'LKR_1000'),
    (r'data\genuine\LKR_1000\1000_G9.jpg', 'LKR_1000'),
    (r'data\genuine\LKR_1000\1000_G10.jpg', 'LKR_1000'),
]

for path, denom in notes:
    if os.path.exists(path):
        evaluate_note(path, denom)
    else:
        print(f"File not found: {path}")
