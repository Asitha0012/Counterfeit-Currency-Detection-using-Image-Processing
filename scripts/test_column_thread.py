import cv2
import numpy as np
import os

def analyze_thread_concentration(folder, denom):
    print(f"\n--- Analyzing {folder} ({denom}) ---")
    files = sorted([f for f in os.listdir(folder) if f.endswith(".jpg")])
    for f in files[:10]:
        img = cv2.imread(os.path.join(folder, f))
        if denom == '500':
            img = cv2.resize(img, (1167, 519))
            thread_crop = img[100:400, 560:630]
        else:
            img = cv2.resize(img, (1165, 455))
            thread_crop = img[100:350, 560:630]
            
        gray_strip = cv2.cvtColor(thread_crop, cv2.COLOR_BGR2GRAY)
        
        # Check green shift
        green_mask = (thread_crop[:, :, 1].astype(np.int16) > thread_crop[:, :, 2].astype(np.int16) + 3) & (gray_strip < 210)
        
        # Compute column-wise ratios
        col_ratios = np.sum(green_mask, axis=0) / green_mask.shape[0]
        max_col_ratio = np.max(col_ratios)
        
        # Let's also check for grayscale
        is_gray = np.all(img[:, :, 0] == img[:, :, 1]) and np.all(img[:, :, 0] == img[:, :, 2])
        
        print(f"  {f:<25} Max Col Ratio: {max_col_ratio:.4f}  Grayscale: {is_gray}")

analyze_thread_concentration("Dataset/500_dataset", "500")
analyze_thread_concentration("Fake Notes/500", "500")
