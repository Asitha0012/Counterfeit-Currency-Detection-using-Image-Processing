import cv2
import numpy as np
import sys

sys.path.append('src')
from features.programmatic_features import PROGRAMMATIC_COORDS

def test_canny(img_path, denom):
    img = cv2.imread(img_path)
    if img is None: return
    x1, y1, x2, y2 = PROGRAMMATIC_COORDS[denom]['security_thread']
    panel = img[y1:y2, x1:x2]
    gray = cv2.cvtColor(panel, cv2.COLOR_BGR2GRAY)
    
    col_sums = np.sum(gray, axis=0)
    best_x = np.argmin(np.convolve(col_sums, np.ones(5)/5, mode='valid')) + 2
    
    edges = cv2.Canny(gray, 30, 100)
    
    kernel = np.ones((5, 5), np.uint8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    valid_rects = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if abs(x + w/2 - best_x) < 20 and w > 5 and h > 10:
            valid_rects.append((x, y, w, h))
            
    max_stacked = 0
    if valid_rects:
        for rx, ry, rw, rh in valid_rects:
            stacked = sum(1 for ox, oy, ow, oh in valid_rects if abs(ox - rx) <= 15)
            if stacked > max_stacked:
                max_stacked = stacked
                
    print(f"{img_path}: Segments {max_stacked}")

test_canny(r'data\genuine\LKR_1000\1000_G6.jpg', 'LKR_1000')
test_canny(r'data\genuine\LKR_1000\1000_G1.jpg', 'LKR_1000')
test_canny(r'data\counterfeit\LKR_1000\1000_F2.png', 'LKR_1000')
