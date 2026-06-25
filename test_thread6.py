import cv2
import numpy as np
import sys

sys.path.append('src')
from features.programmatic_features import PROGRAMMATIC_COORDS

def test_thread_rects(img_path, denom):
    img = cv2.imread(img_path)
    if img is None:
        return
    x1, y1, x2, y2 = PROGRAMMATIC_COORDS[denom]['security_thread']
    panel = img[y1:y2, x1:x2]
    gray = cv2.cvtColor(panel, cv2.COLOR_BGR2GRAY)
    
    # Adaptive threshold
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 21, 10)
    
    kernel = np.ones((5, 5), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    raw_rects = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w > 5 and h > 5:
            raw_rects.append((x, y, w, h))
            
    print(f"\n{img_path}: found {len(raw_rects)} raw rects")
    
    # Find the best vertical alignment
    max_stacked = 0
    best_x = -1
    best_rects = []
    
    for x, y, w, h in raw_rects:
        stacked_rects = [r for r in raw_rects if abs(r[0] - x) <= 15]
        if len(stacked_rects) > max_stacked:
            max_stacked = len(stacked_rects)
            best_x = x
            best_rects = stacked_rects
            
    # Merge overlapping vertically
    best_rects = sorted(best_rects, key=lambda r: r[1])
    merged = True
    while merged:
        merged = False
        used = set()
        for i in range(len(best_rects)):
            if i in used: continue
            rx, ry, rw, rh = best_rects[i]
            
            merge_idx = -1
            for j in range(i + 1, len(best_rects)):
                if j in used: continue
                ox, oy, ow, oh = best_rects[j]
                
                gap = oy - (ry + rh)
                if gap <= 15:
                    merge_idx = j
                    break
                    
            if merge_idx != -1:
                ox, oy, ow, oh = best_rects[merge_idx]
                min_x = min(rx, ox)
                min_y = min(ry, oy)
                max_x = max(rx + rw, ox + ow)
                max_y = max(ry + rh, oy + oh)
                best_rects[i] = (min_x, min_y, max_x - min_x, max_y - min_y)
                used.add(merge_idx)
                merged = True
                break
                
        if merged:
            best_rects = [best_rects[k] for k in range(len(best_rects)) if k not in used]
            best_rects = sorted(best_rects, key=lambda r: r[1])
            
    valid_rects = [r for r in best_rects if r[2] > 10 and r[3] > 20]
    
    print(f"Valid stacked rects: {len(valid_rects)}")
    if valid_rects:
        avg_w = np.mean([r[2] for r in valid_rects])
        measured_mm = avg_w / 7.3
        print(f"Measured mm: {measured_mm:.2f}")

test_thread_rects(r'data\genuine\LKR_1000\1000_G6.jpg', 'LKR_1000')
test_thread_rects(r'data\counterfeit\LKR_1000\1000_F2.png', 'LKR_1000')

