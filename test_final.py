import cv2
import numpy as np
import sys

sys.path.append('src')
from align_note import align_note
from features.programmatic_features import PROGRAMMATIC_COORDS

def test_final(img_path, denom):
    img = cv2.imread(img_path)
    aligned = align_note(img, denom)
    if aligned is None:
        return
        
    x1, y1, x2, y2 = (50, 120, 290, 200) # widen crop to x1=50 to guarantee n1 is visible
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
        
    # Baseline filter FIRST
    # We find the median Y of the largest 5 rects to establish the baseline
    largest_temp = sorted(rects, key=lambda r: r[2] * r[3], reverse=True)[:5]
    med_y = np.median([r[1] + r[3]/2 for r in largest_temp])
    
    # Keep only rects that are on this horizontal line
    aligned_rects = [r for r in rects if abs((r[1] + r[3]/2) - med_y) < 20]
    
    # Now take the 10 largest from the aligned ones
    final_rects = sorted(aligned_rects, key=lambda r: r[2] * r[3], reverse=True)[:10]
    final_rects = sorted(final_rects, key=lambda r: r[0])
    
    print(f"\n{img_path}:")
    print(f"Final rects ({len(final_rects)}):")
    if len(final_rects) > 0:
        heights = [r[3] for r in final_rects]
        print(f"Heights: {heights}")
        increases = 0
        for i in range(1, len(heights)):
            if heights[i] >= heights[i-1] - 3:
                increases += 1
        print(f"Ascend: {increases}/{len(heights)-1}")

test_final(r'data\genuine\LKR_1000\1000_G6.jpg', 'LKR_1000')
test_final(r'data\genuine\LKR_1000\1000_G1.jpg', 'LKR_1000')
test_final(r'data\genuine\LKR_1000\1000_G3.jpg', 'LKR_1000')
test_final(r'data\counterfeit\LKR_1000\1000_F4.png', 'LKR_1000')
