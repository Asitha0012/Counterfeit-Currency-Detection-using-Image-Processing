import cv2
import numpy as np
import os

print("--- Location of maximum green ratio column ---")
for f in sorted(os.listdir("Dataset/500_dataset")):
    if f.endswith(".jpg"):
        img = cv2.imread(os.path.join("Dataset/500_dataset", f))
        img = cv2.resize(img, (1167, 519))
        thread_crop = img[100:400, 560:630]
        gray_strip = cv2.cvtColor(thread_crop, cv2.COLOR_BGR2GRAY)
        
        green_mask = (thread_crop[:, :, 1].astype(np.int16) > thread_crop[:, :, 2].astype(np.int16) + 3) & (gray_strip < 210)
        col_ratios = np.sum(green_mask, axis=0) / green_mask.shape[0]
        
        max_idx = np.argmax(col_ratios) if np.max(col_ratios) > 0 else -1
        if max_idx != -1:
            actual_col = 560 + max_idx
            print(f"{f}: max col at {actual_col} with ratio {col_ratios[max_idx]:.4f}")
        else:
            print(f"{f}: no green pixels detected")
