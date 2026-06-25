import cv2
import numpy as np
import sys

sys.path.append('src')
from features.programmatic_features import PROGRAMMATIC_COORDS

img = cv2.imread(r'data\counterfeit\LKR_1000\1000_F2.png')
x1, y1, x2, y2 = PROGRAMMATIC_COORDS['LKR_1000']['security_thread']
panel = img[y1:y2, x1:x2]

gray = cv2.cvtColor(panel, cv2.COLOR_BGR2GRAY)

# Apply vertical edge detection
sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
abs_sobel_x = np.absolute(sobel_x)
sobel_8u = np.uint8(abs_sobel_x)

# Project edges vertically
col_edge_sums = np.sum(sobel_8u, axis=0)

# The thread should have two strong vertical edges (left and right), or one strong dark line.
# Let's blur vertically and threshold using a local adaptive threshold instead.
blurred = cv2.GaussianBlur(gray, (5, 51), 0)
thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 21, 15)

kernel = np.ones((5, 5), np.uint8)
thresh_clean = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

# Save images to see what they look like
cv2.imwrite('f2_thread_adaptive.png', thresh_clean)

# Find col projection on thresh_clean
col_sums = np.sum(thresh_clean, axis=0)
best_x = np.argmax(col_sums)
print(f"Best X: {best_x}")

# draw rects in best_x +/- 15
contours, _ = cv2.findContours(thresh_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
for c in contours:
    x, y, w, h = cv2.boundingRect(c)
    if abs(x + w/2 - best_x) < 15 and w > 5 and h > 5:
        print(f"  x={x}, y={y}, w={w}, h={h}")
