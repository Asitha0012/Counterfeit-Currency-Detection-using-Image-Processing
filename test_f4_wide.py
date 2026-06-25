import cv2
import sys

sys.path.append('src')
from align_note import align_note

def test_f4_wide():
    img = cv2.imread(r'data\counterfeit\LKR_1000\1000_F4.png')
    aligned = align_note(img, 'LKR_1000')
    x1, y1, x2, y2 = (50, 120, 290, 200) # Widened x1 from 60 to 50
    panel = aligned[y1:y2, x1:x2]
    
    gray = cv2.cvtColor(panel, cv2.COLOR_BGR2GRAY)
    blurred = cv2.medianBlur(gray, 5)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 10)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    panel_width = panel.shape[1]
    
    rects = []
    print("All contours:")
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w >= 2 and h >= 6 and (w * h) > 15:
            aspect_ratio = float(w) / h
            if 0.15 < aspect_ratio < 2.0:
                print(f"  x={x}, y={y}, w={w}, h={h}, baseline={y+h}")
                rects.append((x, y, w, h))
                
    largest = sorted(rects, key=lambda r: r[2]*r[3], reverse=True)[:10]
    if largest:
        baselines = [r[1] + r[3] for r in largest]
        med_b = sorted(baselines)[len(baselines)//2]
        
        aligned_rects = [r for r in rects if abs((r[1] + r[3]) - med_b) < 8]
        final = sorted(aligned_rects, key=lambda r: r[0])
        print(f"\nMedian Baseline: {med_b}")
        print(f"Final aligned characters ({len(final)}):")
        for r in final:
            print(f"  x={r[0]}, y={r[1]}, w={r[2]}, h={r[3]}, baseline={r[1]+r[3]}")

test_f4_wide()
