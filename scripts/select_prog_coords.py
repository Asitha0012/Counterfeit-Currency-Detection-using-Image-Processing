import cv2
import sys

sys.path.append('src')
from align_note import align_note

def main():
    denom = 'LKR_500'
    g1_path = rf"data\genuine\{denom}\500_G1.jpg"
    img = cv2.imread(g1_path)
    aligned_g1 = align_note(img, denom)
    
    if aligned_g1 is None:
        print("Failed to align 500_G1.jpg!")
        return
        
    features = [
        'blind_dots', 
        'asymmetric_serial', 
        'vertical_red_serial', 
        'security_thread', 
        'edge_lines'
    ]
    
    coords = {}
    print("Please select the bounding boxes for the programmatic features.")
    
    for f in features:
        window_name = f"Select {f} (Press ENTER to confirm)"
        r = cv2.selectROI(window_name, aligned_g1, showCrosshair=True, fromCenter=False)
        cv2.destroyWindow(window_name)
        
        x, y, bw, bh = r
        if bw == 0 or bh == 0:
            print(f"Skipped {f}")
            continue
            
        coords[f] = (x, y, x + bw, y + bh)
        
    print("\n=================================")
    print("PROGRAMMATIC_COORDS['LKR_500'] = {")
    for f, (x1, y1, x2, y2) in coords.items():
        print(f"    '{f}': ({x1}, {y1}, {x2}, {y2}),")
    print("}")
    print("=================================\n")
    print("Please copy the dictionary above and provide it to Antigravity!")

if __name__ == '__main__':
    main()
