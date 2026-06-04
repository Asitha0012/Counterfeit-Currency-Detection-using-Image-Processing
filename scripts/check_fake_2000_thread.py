import cv2
import numpy as np
import os

print("--- Fake/Simulated 2000 Notes ---")
for f in sorted(os.listdir("Fake Notes/2000")):
    if f.endswith(".jpg"):
        img = cv2.imread(os.path.join("Fake Notes/2000", f))
        img_res = cv2.resize(img, (1165, 455))
        thread_crop = img_res[100:350, 560:630]
        
        max_ratio_blue = 0.0
        max_ratio_green = 0.0
        for col in range(thread_crop.shape[1] - 15):
            strip = thread_crop[:, col:col+15]
            gray_strip = cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY)
            
            # Blue thread check
            blue_mask = (strip[:, :, 0].astype(np.int16) > strip[:, :, 1].astype(np.int16) + 15) & (gray_strip < 210)
            ratio_b = np.sum(blue_mask) / blue_mask.size
            if ratio_b > max_ratio_blue:
                max_ratio_blue = ratio_b
                
            # Green thread check
            green_mask = (strip[:, :, 1].astype(np.int16) > strip[:, :, 2].astype(np.int16) + 3) & (gray_strip < 210)
            ratio_g = np.sum(green_mask) / green_mask.size
            if ratio_g > max_ratio_green:
                max_ratio_green = ratio_g
                
        print(f"{f}: max_ratio_blue={max_ratio_blue:.4f}, max_ratio_green={max_ratio_green:.4f}")
