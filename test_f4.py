import cv2
import sys

sys.path.append('src')
from align_note import align_note
from features.programmatic_features import PROGRAMMATIC_COORDS

def test_f4():
    img = cv2.imread(r'data\counterfeit\LKR_1000\1000_F4.png')
    aligned = align_note(img, 'LKR_1000')
    x1, y1, x2, y2 = PROGRAMMATIC_COORDS['LKR_1000']['asymmetric_serial']
    panel = aligned[y1:y2, x1:x2]
    
    gray = cv2.cvtColor(panel, cv2.COLOR_BGR2GRAY)
    blurred = cv2.medianBlur(gray, 5)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 10)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    panel_width = panel.shape[1]
    
    print("All contours:")
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        print(f"  x={x}, y={y}, w={w}, h={h}, area={w*h}, aspect={w/h if h else 0}")
        if w >= 2 and h >= 6 and (w * h) > 15:
            aspect_ratio = float(w) / h
            if 0.15 < aspect_ratio < 2.0:
                if x > 2 and (x + w) < (panel_width - 2):
                    print("    -> ACCEPTED!")

test_f4()
