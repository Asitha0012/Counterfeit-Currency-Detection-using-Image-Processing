import cv2
import numpy as np
import os

print("--- 2000 Thread Grayscale Valley Analysis ---")
for f in sorted(os.listdir("Dataset/2000_dataset")):
    if f.endswith(".jpg"):
        img_gen = cv2.imread(os.path.join("Dataset/2000_dataset", f))
        img_res_gen = cv2.resize(img_gen, (1165, 455))
        crop_gen = cv2.cvtColor(img_res_gen[100:350, 560:630], cv2.COLOR_BGR2GRAY)
        col_means_gen = np.mean(crop_gen, axis=0)
        diff_gen = np.mean(col_means_gen) - np.min(col_means_gen)
        
        # Fake
        fake_path = os.path.join("Fake Notes/2000", f"sim_nothread_{f}")
        diff_fake = -1.0
        if os.path.exists(fake_path):
            img_fake = cv2.imread(fake_path)
            img_res_fake = cv2.resize(img_fake, (1165, 455))
            crop_fake = cv2.cvtColor(img_res_fake[100:350, 560:630], cv2.COLOR_BGR2GRAY)
            col_means_fake = np.mean(crop_fake, axis=0)
            diff_fake = np.mean(col_means_fake) - np.min(col_means_fake)
            
        print(f"{f}: Genuine Diff={diff_gen:.1f}, Fake Diff={diff_fake:.1f}")
