import cv2
import numpy as np
import os

def check_color_profile(img, denom):
    if denom == '500':
        center_crop = img[100:400, 200:900]
        hsv = cv2.cvtColor(center_crop, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(center_crop, cv2.COLOR_BGR2GRAY)
        mask = (gray > 15) & (gray < 252)
        if np.sum(mask) == 0:
            return "No mask pixels"
            
        h_vals = hsv[:, :, 0][mask]
        s_vals = hsv[:, :, 1][mask]
        
        h_rad = (h_vals.astype(np.float32) * 2.0) * (np.pi / 180.0)
        avg_x = np.mean(np.cos(h_rad))
        avg_y = np.mean(np.sin(h_rad))
        avg_hue_deg = np.arctan2(avg_y, avg_x) * (180.0 / np.pi)
        if avg_hue_deg < 0:
            avg_hue_deg += 360.0
        avg_hue_opencv = avg_hue_deg / 2.0
        avg_sat = np.mean(s_vals)
        
        base_color_passed = (5.0 <= avg_hue_opencv <= 45.0) and (2.0 <= avg_sat <= 60.0)
        
        # Thread check
        thread_crop = img[100:400, 560:630]
        max_ratio = 0.0
        for col in range(thread_crop.shape[1] - 15):
            strip = thread_crop[:, col:col+15]
            gray_strip = cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY)
            shift_mask = (strip[:, :, 1].astype(np.int16) > strip[:, :, 2].astype(np.int16) + 3) & (gray_strip < 210)
            ratio = np.sum(shift_mask) / shift_mask.size
            if ratio > max_ratio:
                max_ratio = ratio
        thread_passed = (max_ratio > 0.001)
        
        return f"Hue: {avg_hue_opencv:.1f} (passed={5.0 <= avg_hue_opencv <= 45.0}), Sat: {avg_sat:.1f} (passed={2.0 <= avg_sat <= 60.0}), Thread Ratio: {max_ratio:.4f} (passed={thread_passed})"
    else:
        center_crop = img[100:350, 200:900]
        hsv = cv2.cvtColor(center_crop, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(center_crop, cv2.COLOR_BGR2GRAY)
        mask = (gray > 15) & (gray < 252)
        if np.sum(mask) == 0:
            return "No mask pixels"
            
        h_vals = hsv[:, :, 0][mask]
        s_vals = hsv[:, :, 1][mask]
        
        h_rad = (h_vals.astype(np.float32) * 2.0) * (np.pi / 180.0)
        avg_x = np.mean(np.cos(h_rad))
        avg_y = np.mean(np.sin(h_rad))
        avg_hue_deg = np.arctan2(avg_y, avg_x) * (180.0 / np.pi)
        if avg_hue_deg < 0:
            avg_hue_deg += 360.0
        avg_hue_opencv = avg_hue_deg / 2.0
        avg_sat = np.mean(s_vals)
        
        base_color_passed = (130.0 <= avg_hue_opencv <= 178.0) and (5.0 <= avg_sat <= 95.0)
        
        thread_crop = img[100:350, 560:630]
        max_ratio = 0.0
        for col in range(thread_crop.shape[1] - 15):
            strip = thread_crop[:, col:col+15]
            gray_strip = cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY)
            shift_mask = (strip[:, :, 1].astype(np.int16) > strip[:, :, 2].astype(np.int16) + 3) & (gray_strip < 210)
            ratio = np.sum(shift_mask) / shift_mask.size
            if ratio > max_ratio:
                max_ratio = ratio
        thread_passed = (max_ratio > 0.001)
        
        return f"Hue: {avg_hue_opencv:.1f} (passed={130.0 <= avg_hue_opencv <= 178.0}), Sat: {avg_sat:.1f} (passed={5.0 <= avg_sat <= 95.0}), Thread Ratio: {max_ratio:.4f} (passed={thread_passed})"

print("--- 500 Genuine ---")
for f in sorted(os.listdir("Dataset/500_dataset")):
    if f.endswith(".jpg"):
        img = cv2.imread(os.path.join("Dataset/500_dataset", f))
        img = cv2.resize(img, (1167, 519))
        print(f"{f}: {check_color_profile(img, '500')}")

print("\n--- 2000 Genuine ---")
for f in sorted(os.listdir("Dataset/2000_dataset")):
    if f.endswith(".jpg"):
        img = cv2.imread(os.path.join("Dataset/2000_dataset", f))
        img = cv2.resize(img, (1165, 455))
        print(f"{f}: {check_color_profile(img, '2000')}")
