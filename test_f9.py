import cv2
import numpy as np
import sys

sys.path.append('src')
from align_note import align_note
from features.programmatic_features import PROGRAMMATIC_COORDS

def test_asymmetric(img_path, denom):
    img = cv2.imread(img_path)
    aligned = align_note(img, denom)
    if aligned is None:
        print("Alignment failed")
        return
        
    x1, y1, x2, y2 = PROGRAMMATIC_COORDS[denom]['asymmetric_serial']
    panel = aligned[y1:y2, x1:x2]
    
    gray = cv2.cvtColor(panel, cv2.COLOR_BGR2GRAY)
    blurred = cv2.medianBlur(gray, 5)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 10)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    rects = []
    panel_width = panel.shape[1]
    
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w >= 2 and h >= 6 and (w * h) > 15:
            aspect_ratio = float(w) / h
            if 0.15 < aspect_ratio < 2.0:
                if x > 2 and (x + w) < (panel_width - 2):
                    rects.append((x, y, w, h))
                    
    print(f"\n{img_path}:")
    print(f"Total rects found: {len(rects)}")
    
    largest_rects = sorted(rects, key=lambda r: r[2] * r[3], reverse=True)[:10]
    if largest_rects:
        centers_y = [r[1] + r[3]/2 for r in largest_rects]
        median_y = np.median(centers_y)
        
        aligned_rects = []
        for r in rects:
            cy = r[1] + r[3]/2
            if abs(cy - median_y) < 15: # tight tolerance
                aligned_rects.append(r)
                
        final_rects = sorted(aligned_rects, key=lambda r: r[2] * r[3], reverse=True)[:10]
        final_rects = sorted(final_rects, key=lambda r: r[0])
        
        print(f"Final rects count: {len(final_rects)}")
        if len(final_rects) > 0:
            heights = [r[3] for r in final_rects]
            print(f"Heights: {heights}")
            increases = 0
            for i in range(1, len(heights)):
                if heights[i] >= heights[i-1] - 3:
                    increases += 1
            print(f"Ascend: {increases}/{len(heights)-1}")
    else:
        print("No rects found.")

test_asymmetric(r'data\genuine\LKR_1000\1000_G6.jpg', 'LKR_1000')
test_asymmetric(r'data\counterfeit\LKR_1000\1000_F4.png', 'LKR_1000')
test_asymmetric(r'data\genuine\LKR_1000\1000_G1.jpg', 'LKR_1000')
