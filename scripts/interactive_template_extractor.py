import cv2
import os
import sys
import numpy as np

sys.path.append('src')
from align_note import align_note

def main():
    denom = 'LKR_500'
    base_dir = r"data\templates"
    denom_dir = os.path.join(base_dir, denom)
    
    # Create the 7 folders
    for i in [7]:
        os.makedirs(os.path.join(denom_dir, f"Feature {i}"), exist_ok=True)
        
    print("Folders created.")
    
    # Load and align G1
    g1_path = rf"data\genuine\{denom}\500_G1.jpg"
    img = cv2.imread(g1_path)
    aligned_g1 = align_note(img, denom)
    
    if aligned_g1 is None:
        print("Failed to align 500_G1.jpg!")
        return
        
    h, w = aligned_g1.shape[:2]
    
    feature_coords = {}
    print("Please select Feature 7 in the window.")
    
    for i in [7]:
        window_name = f"Select Feature {i} (Press ENTER or SPACE to confirm)"
        r = cv2.selectROI(window_name, aligned_g1, showCrosshair=True, fromCenter=False)
        cv2.destroyWindow(window_name)
        
        x, y, bw, bh = r
        if bw == 0 or bh == 0:
            print(f"Skipped Feature {i}")
            continue
            
        feature_coords[i] = (x, y, bw, bh)
        
        # Calculate percentages
        px1 = x / w
        py1 = y / h
        px2 = (x + bw) / w
        py2 = (y + bh) / h
        
        print(f"Feature {i}: {px1:.3f}, {py1:.3f}, {px2:.3f}, {py2:.3f}")
        
    # Now extract from all G1 to G10
    print("\nExtracting from all 10 genuine notes...")
    for n in range(1, 11):
        note_path = rf"data\genuine\{denom}\500_G{n}.jpg"
        if not os.path.exists(note_path):
            continue
            
        img_n = cv2.imread(note_path)
        aligned_n = align_note(img_n, denom)
        if aligned_n is None:
            continue
            
        for fid, (x, y, bw, bh) in feature_coords.items():
            crop = aligned_n[y:y+bh, x:x+bw]
            out_path = os.path.join(denom_dir, f"Feature {fid}", f"{n}.jpg")
            cv2.imwrite(out_path, crop)
            
    print("\nDone! All templates extracted into data/templates/LKR_500/Feature X")
    print("Please copy the percentage coordinates printed above and provide them to Antigravity!")

if __name__ == '__main__':
    main()
