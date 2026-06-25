import cv2
import numpy as np
import sys

sys.path.append('src')
from features.programmatic_features import PROGRAMMATIC_COORDS

img = cv2.imread(r'data\counterfeit\LKR_1000\1000_F2.png')
x1, y1, x2, y2 = PROGRAMMATIC_COORDS['LKR_1000']['security_thread']
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

print(f"Raw rects found: {len(raw_rects)}")
for r in raw_rects:
    print(f"  x={r[0]}, y={r[1]}, w={r[2]}, h={r[3]}")

# Let's see what the actual thread segments look like.
