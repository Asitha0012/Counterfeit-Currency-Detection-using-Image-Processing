import cv2
import sys
sys.path.append('src')
from align_note import align_note
from features.programmatic_features import PROGRAMMATIC_COORDS

def check_contours(img_path):
    img = cv2.imread(img_path)
    aligned = align_note(img, 'LKR_1000')
    if aligned is None: return
    x1, y1, x2, y2 = PROGRAMMATIC_COORDS['LKR_1000']['asymmetric_serial']
    panel = aligned[y1:y2, x1:x2]
    
    gray = cv2.cvtColor(panel, cv2.COLOR_BGR2GRAY)
    blurred = cv2.medianBlur(gray, 5)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 10)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f"\n{img_path}")
    print(f"Crop width: {panel.shape[1]}")
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w >= 2 and h >= 6 and (w*h) > 15:
            print(f"Rect: x={x}, y={y}, w={w}, h={h}, area={w*h}")

check_contours(r'data\genuine\LKR_1000\1000_G6.jpg')
check_contours(r'data\genuine\LKR_1000\1000_G1.jpg')
