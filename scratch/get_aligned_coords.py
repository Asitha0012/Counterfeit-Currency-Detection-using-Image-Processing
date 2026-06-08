import cv2
import sys
sys.path.append('src')
from align_note import align_note

def get_coords(denom, prefix):
    print(f"\n--- Finding Asymmetric Serial Coordinates for {denom} ---")
    img_path = f'data/genuine/{denom}/Scanned_20260608-{prefix}-01.jpg'
    raw_img = cv2.imread(img_path)
    if raw_img is None: return
        
    img = align_note(raw_img, denom)

    print(f"\nDraw a tight box around the 15 BLACK EDGE LINES on the right side of the note and press ENTER.")
    window_name = f"Select Edge Lines - {denom}"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    roi = cv2.selectROI(window_name, img, showCrosshair=True, fromCenter=False)
    cv2.destroyAllWindows()
    if roi[2] > 0: print(f"✅ EDGE LINES Coordinates: ({roi[0]}, {roi[1]}, {roi[0]+roi[2]}, {roi[1]+roi[3]})")

if __name__ == "__main__":
    get_coords('LKR_1000', '1322')
    get_coords('LKR_5000', '1326')
