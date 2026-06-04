import cv2
import numpy as np

img = cv2.imread("Dataset/500_dataset/500_s4.jpg")
print("Channels are identical:", np.all(img[:, :, 0] == img[:, :, 1]) and np.all(img[:, :, 0] == img[:, :, 2]))
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
print("HSV shape:", hsv.shape)
print("Max Saturation:", np.max(hsv[:, :, 1]))
print("Mean Saturation:", np.mean(hsv[:, :, 1]))
