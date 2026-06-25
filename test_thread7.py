import cv2
import numpy as np
import sys

sys.path.append('src')
from features.programmatic_features import PROGRAMMATIC_COORDS

def test_best(img_path, denom):
    img = cv2.imread(img_path)
    if img is None: return
    x1, y1, x2, y2 = PROGRAMMATIC_COORDS[denom]['security_thread']
    panel = img[y1:y2, x1:x2]
    gray = cv2.cvtColor(panel, cv2.COLOR_BGR2GRAY)
    
    # 1. Find aligned line using column projection
    col_sums = np.sum(gray, axis=0)
    window = 5
    smoothed_cols = np.convolve(col_sums, np.ones(window)/window, mode='valid')
    best_x = np.argmin(smoothed_cols) + window//2
    
    # 2. Extract narrow strip to isolate the thread vertically
    # Thread width is typically ~20 pixels. So +/- 15 is safe
    strip_x1 = max(0, best_x - 15)
    strip_x2 = min(gray.shape[1], best_x + 15)
    strip = gray[:, strip_x1:strip_x2]
    
    # 3. Use OTSU thresholding locally on the strip!
    _, strip_thresh = cv2.threshold(strip, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Morphological open to remove noise
    kernel = np.ones((5, 5), np.uint8)
    strip_thresh = cv2.morphologyEx(strip_thresh, cv2.MORPH_OPEN, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(strip_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    valid_rects = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w >= 5 and h >= 10:
            valid_rects.append((x, y, w, h))
            
    # Count stacked segments
    max_stacked = 0
    if valid_rects:
        for rx, ry, rw, rh in valid_rects:
            stacked = sum(1 for ox, oy, ow, oh in valid_rects if abs(ox - rx) <= 15)
            if stacked > max_stacked:
                max_stacked = stacked
                
    if valid_rects:
        avg_w = np.mean([rw for rx, ry, rw, rh in valid_rects])
        measured_mm = avg_w / 7.3
    else:
        measured_mm = 0.0
        
    print(f"{img_path}:")
    print(f"  Segments: {max_stacked}")
    print(f"  Measured mm: {measured_mm:.2f}")

test_best(r'data\genuine\LKR_1000\1000_G1.jpg', 'LKR_1000')
test_best(r'data\genuine\LKR_1000\1000_G3.jpg', 'LKR_1000')
test_best(r'data\genuine\LKR_1000\1000_G6.jpg', 'LKR_1000')
test_best(r'data\counterfeit\LKR_1000\1000_F2.png', 'LKR_1000')

