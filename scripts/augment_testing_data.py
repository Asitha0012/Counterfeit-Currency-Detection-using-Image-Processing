import cv2
import numpy as np
import os
import random
import glob
import shutil

def augment_light(image):
    """Light augmentation that simulates normal camera variations (lighting)."""
    augmented = image.copy()
    
    # Minor Brightness (-10% to +10%)
    hsv = cv2.cvtColor(augmented, cv2.COLOR_BGR2HSV)
    hsv = np.array(hsv, dtype=np.float64)
    brightness_factor = random.uniform(0.9, 1.1) 
    hsv[:,:,2] = hsv[:,:,2] * brightness_factor
    hsv[:,:,2][hsv[:,:,2] > 255] = 255
    hsv = np.array(hsv, dtype=np.uint8)
    
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def generate_datasets(genuine_dir, counterfeit_dir, base_output_dir, denom):
    gen_out_dir = os.path.join(base_output_dir, "Genuine", denom)
    fake_out_dir = os.path.join(base_output_dir, "Fake", denom)
    
    os.makedirs(gen_out_dir, exist_ok=True)
    os.makedirs(fake_out_dir, exist_ok=True)
        
    # --- 1. Process Genuine Notes ---
    gen_paths = glob.glob(os.path.join(genuine_dir, '*.jpg')) + glob.glob(os.path.join(genuine_dir, '*.png'))
    if gen_paths:
        for img_path in gen_paths:
            filename = os.path.basename(img_path)
            name, ext = os.path.splitext(filename)
            image = cv2.imread(img_path)
            if image is None: continue
            
            for i in range(50):
                cv2.imwrite(os.path.join(gen_out_dir, f"{name}_gen_{i+1}{ext}"), augment_light(image))

    # --- 2. Process ACTUAL Counterfeit Notes ---
    fake_paths = glob.glob(os.path.join(counterfeit_dir, '*.jpg')) + glob.glob(os.path.join(counterfeit_dir, '*.png')) + glob.glob(os.path.join(counterfeit_dir, '*.jpeg'))
    if fake_paths:
        for img_path in fake_paths:
            filename = os.path.basename(img_path)
            name, ext = os.path.splitext(filename)
            image = cv2.imread(img_path)
            if image is None: continue
            
            for i in range(50):
                cv2.imwrite(os.path.join(fake_out_dir, f"{name}_fake_{i+1}.jpg"), augment_light(image))

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_base = os.path.join(base_dir, "data", "augmented_testing")
    
    # Clean up the old directory structure
    if os.path.exists(output_base):
        shutil.rmtree(output_base)
        
    print("--- Generating LKR 500 Datasets using REAL Counterfeits ---")
    gen_500 = os.path.join(base_dir, "data", "genuine", "LKR_500")
    fake_500 = os.path.join(base_dir, "data", "counterfeit", "LKR_500")
    generate_datasets(gen_500, fake_500, output_base, "LKR_500")
    
    print("--- Generating LKR 1000 Datasets using REAL Counterfeits ---")
    gen_1000 = os.path.join(base_dir, "data", "genuine", "LKR_1000")
    fake_1000 = os.path.join(base_dir, "data", "counterfeit", "LKR_1000")
    generate_datasets(gen_1000, fake_1000, output_base, "LKR_1000")
    
    print("--- Generating LKR 5000 Datasets using REAL Counterfeits ---")
    gen_5000 = os.path.join(base_dir, "data", "genuine", "LKR_5000")
    fake_5000 = os.path.join(base_dir, "data", "counterfeit", "LKR_5000")
    generate_datasets(gen_5000, fake_5000, output_base, "LKR_5000")
    
    print("Done! Generated EXACTLY 50 augmented images per input note.")
