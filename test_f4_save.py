import cv2
import sys

sys.path.append('src')
from align_note import align_note
from features.programmatic_features import PROGRAMMATIC_COORDS

def test_f4_save():
    img = cv2.imread(r'data\counterfeit\LKR_1000\1000_F4.png')
    aligned = align_note(img, 'LKR_1000')
    x1, y1, x2, y2 = PROGRAMMATIC_COORDS['LKR_1000']['asymmetric_serial']
    panel = aligned[y1:y2, x1:x2]
    cv2.imwrite('f4_panel.png', panel)
    
    gray = cv2.cvtColor(panel, cv2.COLOR_BGR2GRAY)
    blurred = cv2.medianBlur(gray, 5)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 10)
    cv2.imwrite('f4_thresh.png', thresh)

test_f4_save()
