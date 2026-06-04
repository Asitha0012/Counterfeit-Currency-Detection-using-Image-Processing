import cv2
import numpy as np
import os

print("--- Grayscale Thread Intensity Analysis ---")
files = ["500_s4.jpg", "500_s5.jpg", "500_s6.jpg", "500_s7.jpg", "500_s8.jpg"]
for f in files:
    # Genuine
    img_gen = cv2.imread(os.path.join("Dataset/500_dataset", f))
    img_gen = cv2.resize(img_gen, (1167, 519))
    crop_gen = cv2.cvtColor(img_gen[100:400, 560:630], cv2.COLOR_BGR2GRAY)
    col_means_gen = np.mean(crop_gen, axis=0)
    
    # Fake
    img_fake = cv2.imread(os.path.join("Fake Notes/500", f"sim_nothread_{f}"))
    img_fake = cv2.resize(img_fake, (1167, 519))
    crop_fake = cv2.cvtColor(img_fake[100:400, 560:630], cv2.COLOR_BGR2GRAY)
    col_means_fake = np.mean(crop_fake, axis=0)
    
    # Let's print the standard deviation or min-max difference of column means
    # A real thread will cause a local drop in intensity (a valley)
    # Let's print the minimum column mean and the average column mean
    print(f"{f}:")
    print(f"  Genuine: min_col_mean={np.min(col_means_gen):.1f}, avg_col_mean={np.mean(col_means_gen):.1f}, diff={np.mean(col_means_gen) - np.min(col_means_gen):.1f}")
    print(f"  Fake:    min_col_mean={np.min(col_means_fake):.1f}, avg_col_mean={np.mean(col_means_fake):.1f}, diff={np.mean(col_means_fake) - np.min(col_means_fake):.1f}")
