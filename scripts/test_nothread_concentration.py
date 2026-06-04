import cv2
import numpy as np
import os

print("--- Thread-Painted Fake 500 Notes ---")
for f in sorted(os.listdir("Fake Notes/500")):
    if "sim_nothread" in f:
        img = cv2.imread(os.path.join("Fake Notes/500", f))
        img = cv2.resize(img, (1167, 519))
        thread_crop = img[100:400, 560:630]
        gray_strip = cv2.cvtColor(thread_crop, cv2.COLOR_BGR2GRAY)
        
        green_mask = (thread_crop[:, :, 1].astype(np.int16) > thread_crop[:, :, 2].astype(np.int16) + 3) & (gray_strip < 210)
        col_ratios = np.sum(green_mask, axis=0) / green_mask.shape[0]
        max_col_ratio = np.max(col_ratios) if len(col_ratios) > 0 else 0.0
        
        # Is it grayscale?
        is_gray = np.all(img[:, :, 0] == img[:, :, 1]) and np.all(img[:, :, 0] == img[:, :, 2])
        
        print(f"  {f:<25} Max Col Ratio: {max_col_ratio:.4f}  Grayscale: {is_gray}")
