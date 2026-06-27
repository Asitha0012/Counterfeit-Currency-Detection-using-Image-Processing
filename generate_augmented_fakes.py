import cv2
import os
import glob
import numpy as np
import random
import shutil

def change_brightness(img, value):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v = cv2.add(v, value)
    final_hsv = cv2.merge((h, s, v))
    return cv2.cvtColor(final_hsv, cv2.HSV_BGR)

def change_contrast(img, alpha):
    return cv2.convertScaleAbs(img, alpha=alpha, beta=0)

def change_hsv(img, h_shift, s_shift, v_shift):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.int16)
    h, s, v = cv2.split(hsv)
    h = (h + h_shift) % 180
    s = np.clip(s + s_shift, 0, 255)
    v = np.clip(v + v_shift, 0, 255)
    final_hsv = cv2.merge((h, s, v)).astype(np.uint8)
    return cv2.cvtColor(final_hsv, cv2.HSV_BGR)

def add_gaussian_noise(img, sigma):
    noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
    noisy = cv2.add(img.astype(np.float32), noise)
    return np.clip(noisy, 0, 255).astype(np.uint8)

def process_denomination(denom, num_fakes_expected, avoid_files=[]):
    in_dir = f'data/counterfeit/{denom}'
    out_dir_base = f'data/augmented_testing/fake/{denom}'
    
    # Delete existing
    if os.path.exists(out_dir_base):
        shutil.rmtree(out_dir_base)
    
    folders = ['brightness', 'contrast', 'sigma', 'hsv']
    for folder in folders:
        os.makedirs(os.path.join(out_dir_base, folder), exist_ok=True)
        
    fake_files = [f for f in os.listdir(in_dir) if f.endswith(('.jpg', '.png', '.jpeg')) and f not in avoid_files]
    print(f"{denom}: found {len(fake_files)} source files. Generating 500 augmented fakes...")
    
    if len(fake_files) == 0:
        return
        
    num_per_file = 500 // len(fake_files)
    remainder = 500 % len(fake_files)
    
    count = 0
    for file_idx, f in enumerate(fake_files):
        img_path = os.path.join(in_dir, f)
        img = cv2.imread(img_path)
        if img is None: continue
        
        target_count = num_per_file + (1 if file_idx < remainder else 0)
        
        for i in range(target_count):
            aug_type = random.choice(folders)
            is_extreme = random.random() < 0.2
            
            aug_img = None
            if aug_type == 'brightness':
                if is_extreme:
                    val = random.choice([random.randint(-120, -80), random.randint(80, 120)])
                else:
                    val = random.choice([random.randint(-60, -20), random.randint(20, 60)])
                aug_img = change_brightness(img, val)
                
            elif aug_type == 'contrast':
                if is_extreme:
                    alpha = random.choice([random.uniform(0.3, 0.5), random.uniform(2.5, 3.5)])
                else:
                    alpha = random.choice([random.uniform(0.6, 0.8), random.uniform(1.2, 1.8)])
                aug_img = change_contrast(img, alpha)
                
            elif aug_type == 'sigma':
                if is_extreme:
                    sigma = random.uniform(35, 60)
                else:
                    sigma = random.uniform(10, 25)
                aug_img = add_gaussian_noise(img, sigma)
                
            elif aug_type == 'hsv':
                # HSV changes are wild, no passing range considerations needed
                h_shift = random.randint(-40, 40)
                s_shift = random.randint(-50, 50)
                v_shift = random.randint(-50, 50)
                aug_img = change_hsv(img, h_shift, s_shift, v_shift)
                
            out_name = f"{f.split('.')[0]}_{aug_type}_{i}.jpg"
            out_path = os.path.join(out_dir_base, aug_type, out_name)
            cv2.imwrite(out_path, aug_img)
            count += 1
            
    print(f"{denom}: Generated {count} images.")

if __name__ == '__main__':
    # 500: 8 notes
    process_denomination('LKR_500', 500)
    # 1000: 5 notes, avoid '1000_F5 - Copy.jpg'
    process_denomination('LKR_1000', 500, avoid_files=['1000_F5 - Copy.jpg'])
    # 5000: 6 notes
    process_denomination('LKR_5000', 500)
