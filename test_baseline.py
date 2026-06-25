import cv2
import sys
import numpy as np

sys.path.append('src')
from align_note import align_note
from features.programmatic_features import PROGRAMMATIC_COORDS

def test_baseline(img_path, denom):
    img = cv2.imread(img_path)
    aligned = align_note(img, denom)
    if aligned is None:
        return
        
    x1, y1, x2, y2 = (45, 120, 290, 200) # widen slightly to 45
    panel = aligned[y1:y2, x1:x2]
    
    gray = cv2.cvtColor(panel, cv2.COLOR_BGR2GRAY)
    blurred = cv2.medianBlur(gray, 5)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 10)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    rects = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w >= 2 and h >= 6 and (w * h) > 15:
            aspect_ratio = float(w) / h
            if 0.15 < aspect_ratio < 2.0:
                rects.append((x, y, w, h))
                
    if not rects:
        print(f"{img_path}: Failed to find any rects")
        return
        
    largest = sorted(rects, key=lambda r: r[2]*r[3], reverse=True)[:10]
    baselines = [r[1] + r[3] for r in largest]
    med_b = sorted(baselines)[len(baselines)//2]
    
    aligned_rects = [r for r in rects if abs((r[1] + r[3]) - med_b) <= 8]
    final = sorted(aligned_rects, key=lambda r: r[2]*r[3], reverse=True)[:10]
    final = sorted(final, key=lambda r: r[0])
    
    print(f"\n{img_path}:")
    print(f"Final rects ({len(final)}):")
    if len(final) > 0:
        heights = [r[3] for r in final]
        print(f"Heights: {heights}")
        increases = 0
        for i in range(1, len(heights)):
            if heights[i] >= heights[i-1] - 3:
                increases += 1
        print(f"Ascend: {increases}/{len(heights)-1}")

test_baseline(r'data\genuine\LKR_1000\1000_G6.jpg', 'LKR_1000')
test_baseline(r'data\genuine\LKR_1000\1000_G1.jpg', 'LKR_1000')
test_baseline(r'data\genuine\LKR_1000\1000_G3.jpg', 'LKR_1000')
test_baseline(r'data\counterfeit\LKR_1000\1000_F4.png', 'LKR_1000')

