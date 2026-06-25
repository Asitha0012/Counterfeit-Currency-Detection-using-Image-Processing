import cv2
import numpy as np
import os
import sys

sys.path.append('src')
from features.programmatic_features import PROGRAMMATIC_COORDS

img = cv2.imread(r'data\counterfeit\LKR_1000\1000_F2.png')
x1, y1, x2, y2 = PROGRAMMATIC_COORDS['LKR_1000']['security_thread']
panel = img[y1:y2, x1:x2]

cv2.imwrite('f2_thread.png', panel)

gray = cv2.cvtColor(panel, cv2.COLOR_BGR2GRAY)

mean_brightness = np.mean(gray)
dynamic_thresh = max(30, int(mean_brightness - 50))

_, thresh = cv2.threshold(gray, dynamic_thresh, 255, cv2.THRESH_BINARY_INV)
cv2.imwrite('f2_thread_thresh.png', thresh)

# Column projection
col_sums = np.sum(thresh, axis=0)
best_x = np.argmax(col_sums)

print(f"Mean brightness: {mean_brightness}, Thresh: {dynamic_thresh}")
print(f"Best X from column sums of thresholded image: {best_x}")

# Draw line
vis = panel.copy()
cv2.line(vis, (best_x, 0), (best_x, panel.shape[0]), (0, 0, 255), 1)
cv2.imwrite('f2_thread_line.png', vis)

