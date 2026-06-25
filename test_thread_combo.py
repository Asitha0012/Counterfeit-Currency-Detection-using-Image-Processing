import cv2
import numpy as np
import sys

sys.path.append('src')
from features.programmatic_features import PROGRAMMATIC_COORDS

def test_combo(img_path, denom):
    img = cv2.imread(img_path)
    if img is None: return
    x1, y1, x2, y2 = PROGRAMMATIC_COORDS[denom]['security_thread']
    panel = img[y1:y2, x1:x2]
    gray = cv2.cvtColor(panel, cv2.COLOR_BGR2GRAY)
    
    col_sums = np.sum(gray, axis=0)
    window = 5
    smoothed_cols = np.convolve(col_sums, np.ones(window)/window, mode='valid')
    best_x = np.argmin(smoothed_cols) + window//2
    
    strip = gray[:, max(0, best_x-5) : min(gray.shape[1], best_x+5)]
    row_sums = np.sum(strip, axis=1)
    row_smoothed = np.convolve(row_sums, np.ones(5)/5, mode='same')
    
    valleys = []
    is_in_valley = False
    valley_start = 0
    thresh_val = np.mean(row_smoothed) * 0.95
    
    segments = []
    for y, val in enumerate(row_smoothed):
        if val < thresh_val:
            if not is_in_valley:
                is_in_valley = True
                valley_start = y
        else:
            if is_in_valley:
                is_in_valley = False
                valley_end = y
                if valley_end - valley_start > 5:
                    segments.append((valley_start, valley_end))
                    
    dynamic_thresh = max(30, int(np.mean(gray) - 50))
    widths = []
    for y_s, y_e in segments:
        seg_roi = gray[y_s:y_e, max(0, best_x-15):min(gray.shape[1], best_x+15)]
        _, seg_thresh = cv2.threshold(seg_roi, dynamic_thresh, 255, cv2.THRESH_BINARY_INV)
        
        row_widths = []
        for r in range(seg_thresh.shape[0]):
            non_zero = np.nonzero(seg_thresh[r,:])[0]
            if len(non_zero) > 0:
                w = non_zero[-1] - non_zero[0]
                if w > 2:
                    row_widths.append(w)
        if row_widths:
            widths.append(np.median(row_widths))
            
    print(f"{img_path}:")
    print(f"  Segments: {len(segments)}")
    if widths:
        avg_w = np.mean(widths)
        measured_mm = avg_w / 7.3
        print(f"  Measured mm: {measured_mm:.2f}")

test_combo(r'data\genuine\LKR_1000\1000_G6.jpg', 'LKR_1000')
test_combo(r'data\genuine\LKR_1000\1000_G1.jpg', 'LKR_1000')
test_combo(r'data\counterfeit\LKR_1000\1000_F2.png', 'LKR_1000')
