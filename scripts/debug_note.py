import cv2
import numpy as np
import os

img500 = cv2.imread("Dataset/500_dataset/500_s4.jpg")
print("500_s4 shape:", img500.shape if img500 is not None else "None")
if img500 is not None:
    img500_res = cv2.resize(img500, (1167, 519))
    center_crop = img500_res[100:400, 200:900]
    gray = cv2.cvtColor(center_crop, cv2.COLOR_BGR2GRAY)
    print("gray range:", np.min(gray), np.max(gray))
    mask = (gray > 15) & (gray < 252)
    print("mask sum:", np.sum(mask))

img2000 = cv2.imread("Dataset/2000_dataset/2000_s1.jpg")
print("2000_s1 shape:", img2000.shape if img2000 is not None else "None")
if img2000 is not None:
    img2000_res = cv2.resize(img2000, (1165, 455))
    thread_crop = img2000_res[100:350, 560:630]
    hsv = cv2.cvtColor(thread_crop, cv2.COLOR_BGR2HSV)
    gray_strip = cv2.cvtColor(thread_crop, cv2.COLOR_BGR2GRAY)
    print("2000 thread crop shape:", thread_crop.shape)
    # Check green pixels (where G > R or in HSV green range)
    # Thread green color: usually Hue 35 to 85, Sat > 30, Val > 30
    green_mask = (hsv[:, :, 0] >= 35) & (hsv[:, :, 0] <= 85) & (hsv[:, :, 1] >= 20)
    print("HSV Green mask sum:", np.sum(green_mask))
    
    # Or let's see if G > R + delta
    g_gt_r = (thread_crop[:, :, 1].astype(np.int16) > thread_crop[:, :, 2].astype(np.int16) + 3)
    print("G > R + 3 mask sum:", np.sum(g_gt_r))
    print("gray_strip < 210 sum:", np.sum(gray_strip < 210))
    print("Combined mask sum:", np.sum(g_gt_r & (gray_strip < 210)))
