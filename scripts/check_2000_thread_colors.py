import cv2
import numpy as np
import os

for f in sorted(os.listdir("Dataset/2000_dataset")):
    if f.endswith(".jpg"):
        img = cv2.imread(os.path.join("Dataset/2000_dataset", f))
        img_res = cv2.resize(img, (1165, 455))
        thread_crop = img_res[100:350, 560:630]
        # Let's see if there is any column with a thread
        # In a real 2000 note, where is the thread? Let's check B, G, R values column by column
        max_b_minus_g = -999
        max_g_minus_r = -999
        max_b_minus_r = -999
        for col in range(thread_crop.shape[1]):
            col_strip = thread_crop[:, col]
            # Average B, G, R in this column
            b = np.mean(col_strip[:, 0])
            g = np.mean(col_strip[:, 1])
            r = np.mean(col_strip[:, 2])
            if b - g > max_b_minus_g: max_b_minus_g = b - g
            if g - r > max_g_minus_r: max_g_minus_r = g - r
            if b - r > max_b_minus_r: max_b_minus_r = b - r
        print(f"{f}: max(B-G)={max_b_minus_g:.1f}, max(G-R)={max_g_minus_r:.1f}, max(B-R)={max_b_minus_r:.1f}")
