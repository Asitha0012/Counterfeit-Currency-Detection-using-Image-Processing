import cv2
import sys
import numpy as np

sys.path.append('src')
from align_note import align_note
from features.programmatic_features import verify_security_thread, PROGRAMMATIC_COORDS

def test_proper(img_path, denom):
    img = cv2.imread(img_path)
    aligned, _ = align_note(img, denom)
    if aligned is None:
        print("Alignment failed")
        return
        
    passed, msg, _ = verify_security_thread(aligned, denom)
    print(f"{img_path}: {passed} - {msg}")

test_proper(r'data\genuine\LKR_1000\1000_G6.jpg', 'LKR_1000')
test_proper(r'data\genuine\LKR_1000\1000_G3.jpg', 'LKR_1000')
test_proper(r'data\counterfeit\LKR_1000\1000_F2.png', 'LKR_1000')
