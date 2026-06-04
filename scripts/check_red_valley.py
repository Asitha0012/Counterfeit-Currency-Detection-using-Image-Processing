import cv2
import numpy as np
import os

print("--- 2000 Thread Red Channel Valley Analysis ---")
for f in sorted(os.listdir("Dataset/2000_dataset")):
    if f.endswith(".jpg"):
        img_gen = cv2.imread(os.path.join("Dataset/2000_dataset", f))
        img_res_gen = cv2.resize(img_gen, (1165, 455))
        crop_gen = img_res_gen[100:350, 560:630]
        # Red channel is index 2 in BGR
        red_crop_gen = crop_gen[:, :, 2]
        col_means_gen = np.mean(red_crop_gen, axis=0)
        diff_gen = np.mean(col_means_gen) - np.min(col_means_gen)
        
        # Fake
        fake_path = os.path.join("Fake Notes/2000", f"sim_nothread_{f}")
        diff_fake = -1.0
        if os.path.exists(fake_path):
            img_fake = cv2.imread(fake_path)
            img_res_fake = cv2.resize(img_fake, (1165, 455))
            crop_fake = img_res_fake[100:350, 560:630]
            red_crop_fake = crop_fake[:, :, 2]
            col_means_fake = np.mean(red_crop_fake, axis=0)
            diff_fake = np.mean(col_means_fake) - np.min(col_means_fake)
            
        print(f"{f}: Genuine Red Diff={diff_gen:.1f}, Fake Red Diff={diff_fake:.1f}")
