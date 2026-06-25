import cv2
import numpy as np
import sys

sys.path.append('src')
from features.programmatic_features import PROGRAMMATIC_COORDS

def test_thread_orig_h(img_path, denom):
    img = cv2.imread(img_path)
    if img is None: return
    x1, y1, x2, y2 = PROGRAMMATIC_COORDS[denom]['security_thread']
    panel = img[y1:y2, x1:x2]
    gray = cv2.cvtColor(panel, cv2.COLOR_BGR2GRAY)
    
    mean_brightness = np.mean(gray)
    dynamic_thresh = max(30, int(mean_brightness - 50))
    
    _, thresh = cv2.threshold(gray, dynamic_thresh, 255, cv2.THRESH_BINARY_INV)
    
    kernel = np.ones((5, 5), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    raw_rects = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w > 5 and h > 5:
            raw_rects.append((x, y, w, h))
            
    raw_rects = sorted(raw_rects, key=lambda r: r[1])
    merged = True
    while merged:
        merged = False
        used = set()
        for i in range(len(raw_rects)):
            if i in used: continue
            rx, ry, rw, rh = raw_rects[i]
            merge_idx = -1
            for j in range(i + 1, len(raw_rects)):
                if j in used: continue
                ox, oy, ow, oh = raw_rects[j]
                overlap_x = max(rx, ox) <= min(rx + rw, ox + ow) + 5
                gap = oy - (ry + rh)
                if overlap_x and gap <= 15:
                    merge_idx = j
                    break
            if merge_idx != -1:
                ox, oy, ow, oh = raw_rects[merge_idx]
                min_x = min(rx, ox)
                min_y = min(ry, oy)
                max_x = max(rx + rw, ox + ow)
                max_y = max(ry + rh, oy + oh)
                raw_rects[i] = (min_x, min_y, max_x - min_x, max_y - min_y)
                used.add(merge_idx)
                merged = True
                break
        if merged:
            raw_rects = [raw_rects[k] for k in range(len(raw_rects)) if k not in used]
            raw_rects = sorted(raw_rects, key=lambda r: r[1])
            
    print(f"\n{img_path}:")
    for rx, ry, rw, rh in raw_rects:
        print(f"  w={rw}, h={rh}")

test_thread_orig_h(r'data\genuine\LKR_1000\1000_G6.jpg', 'LKR_1000')
test_thread_orig_h(r'data\counterfeit\LKR_1000\1000_F2.png', 'LKR_1000')
