import cv2
import numpy as np
import os

print("--- B - R difference in 2000 Notes ---")
for f in sorted(os.listdir("Dataset/2000_dataset")):
    if f.endswith(".jpg"):
        img_gen = cv2.imread(os.path.join("Dataset/2000_dataset", f))
        img_res_gen = cv2.resize(img_gen, (1165, 455))
        crop_gen = img_res_gen[100:350, 560:630]
        # Max B - R in any single pixel or column
        max_diff_gen = np.max(crop_gen[:, :, 0].astype(np.int16) - crop_gen[:, :, 2].astype(np.int16))
        
        # Fake
        fake_path = os.path.join("Fake Notes/2000", f"sim_nothread_{f}")
        max_diff_fake = -999
        if os.path.exists(fake_path):
            img_fake = cv2.imread(fake_path)
            img_res_fake = cv2.resize(img_fake, (1165, 455))
            crop_fake = img_res_fake[100:350, 560:630]
            max_diff_fake = np.max(crop_fake[:, :, 0].astype(np.int16) - crop_fake[:, :, 2].astype(np.int16))
            
        print(f"{f}: Genuine Max(B-R)={max_diff_gen}, Fake Max(B-R)={max_diff_fake}")
