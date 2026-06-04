import cv2
import numpy as np
import os

print("--- 500 Dataset ---")
for f in sorted(os.listdir("Dataset/500_dataset")):
    if f.endswith(".jpg"):
        img = cv2.imread(os.path.join("Dataset/500_dataset", f))
        identical = np.all(img[:, :, 0] == img[:, :, 1]) and np.all(img[:, :, 0] == img[:, :, 2])
        print(f"{f}: Grayscale={identical}")

print("\n--- 2000 Dataset ---")
for f in sorted(os.listdir("Dataset/2000_dataset")):
    if f.endswith(".jpg"):
        img = cv2.imread(os.path.join("Dataset/2000_dataset", f))
        identical = np.all(img[:, :, 0] == img[:, :, 1]) and np.all(img[:, :, 0] == img[:, :, 2])
        print(f"{f}: Grayscale={identical}")
