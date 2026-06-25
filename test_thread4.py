import cv2
import numpy as np
import sys

sys.path.append('src')
from features.programmatic_features import PROGRAMMATIC_COORDS

img = cv2.imread(r'data\counterfeit\LKR_1000\1000_F2.png')
x1, y1, x2, y2 = PROGRAMMATIC_COORDS['LKR_1000']['security_thread']
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

mean_strip = np.mean(row_smoothed)
thresh = mean_strip * 0.95

segments = []
for y, val in enumerate(row_smoothed):
    if val < thresh:
        if not is_in_valley:
            is_in_valley = True
            valley_start = y
    else:
        if is_in_valley:
            is_in_valley = False
            valley_end = y
            if valley_end - valley_start > 5:
                segments.append((valley_start, valley_end))

print(f"Found {len(segments)} segments")

widths = []
for s in segments:
    y_s, y_e = s
    seg_roi = gray[y_s:y_e, max(0, best_x-20):min(gray.shape[1], best_x+20)]
    _, seg_thresh = cv2.threshold(seg_roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # measure width of each row and take median
    row_widths = []
    for r in range(seg_thresh.shape[0]):
        non_zero = np.nonzero(seg_thresh[r,:])[0]
        if len(non_zero) > 0:
            w = non_zero[-1] - non_zero[0]
            if w > 2:
                row_widths.append(w)
    
    if row_widths:
        widths.append(np.median(row_widths))

if widths:
    avg_w = np.mean(widths)
    measured_mm = avg_w / 7.3
    print(f"Average width pixels: {avg_w}, Measured mm: {measured_mm}")
else:
    print("No widths found.")

