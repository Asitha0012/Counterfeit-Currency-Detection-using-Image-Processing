import cv2
import numpy as np
import sys

sys.path.append('src')
from features.programmatic_features import PROGRAMMATIC_COORDS

img = cv2.imread(r'data\counterfeit\LKR_1000\1000_F2.png')
x1, y1, x2, y2 = PROGRAMMATIC_COORDS['LKR_1000']['security_thread']
panel = img[y1:y2, x1:x2]
gray = cv2.cvtColor(panel, cv2.COLOR_BGR2GRAY)

# Find the aligned line by doing column-wise sum. The thread is dark, so it should have a local minimum.
# To be robust against shadows, we can use edge detection or just look at the middle of the panel.
col_sums = np.sum(gray, axis=0)

# Smooth the col_sums to avoid noise
window = 5
smoothed_cols = np.convolve(col_sums, np.ones(window)/window, mode='valid')
best_x = np.argmin(smoothed_cols) + window//2

print(f"Best X from darkest column: {best_x}")

# Extract the strip
strip = gray[:, best_x-5 : best_x+5]
row_sums = np.sum(strip, axis=1)

# Smooth row sums
row_smoothed = np.convolve(row_sums, np.ones(5)/5, mode='same')

# Find valleys (dark segments)
valleys = []
is_in_valley = False
valley_start = 0

mean_strip = np.mean(row_smoothed)
thresh = mean_strip * 0.95  # 5% darker than average

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
            if valley_end - valley_start > 5: # min height
                segments.append((valley_start, valley_end))

print(f"Found {len(segments)} segments")
for s in segments:
    print(f"  y_start={s[0]}, y_end={s[1]}, height={s[1]-s[0]}")
    
