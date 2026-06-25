import os
import glob
import sys
import numpy as np
import cv2

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(base_dir, 'src'))

from evaluate_lkr import analyze_lkr_note
import pandas as pd

def apply_rotation(img, angle):
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # Calculate the new bounding box dimensions to prevent clipping
    # This purely expands the canvas without scaling or degrading the note's quality
    abs_cos = abs(M[0, 0])
    abs_sin = abs(M[0, 1])
    bound_w = int(h * abs_sin + w * abs_cos)
    bound_h = int(h * abs_cos + w * abs_sin)
    
    # Adjust translation in the rotation matrix to shift the image into the new canvas
    M[0, 2] += bound_w / 2 - center[0]
    M[1, 2] += bound_h / 2 - center[1]
    
    # Warp affine onto the larger canvas (note remains at 100% original quality)
    return cv2.warpAffine(img, M, (bound_w, bound_h), borderValue=(255, 255, 255))

def apply_illumination(img, shift):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v = cv2.add(v, shift)
    hsv = cv2.merge((h, s, v))
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def apply_noise(img, sigma):
    noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
    noisy_img = cv2.add(img.astype(np.float32), noise)
    return np.clip(noisy_img, 0, 255).astype(np.uint8)

def test_condition(img_data_list, condition_func, param):
    failed = 0
    total = len(img_data_list)
    for img_path, denom in img_data_list:
        img = cv2.imread(img_path)
        if img is None: continue
        
        distorted = condition_func(img, param)
        
        # Save to temp file to run analyze_lkr_note which takes a file path
        tmp_path = os.path.join(os.path.dirname(__file__), "tmp_stress.jpg")
        cv2.imwrite(tmp_path, distorted)
        
        _, robust_verdict, _, _, _, _ = analyze_lkr_note(tmp_path, denom)
        if not robust_verdict:
            failed += 1
            
    if os.path.exists(tmp_path):
        os.remove(tmp_path)
        
    return (failed / total) * 100

def run_stress_test():
    gen_1000 = glob.glob(os.path.join(base_dir, "data", "genuine", "LKR_1000", '*.jpg'))[:10]
    gen_5000 = glob.glob(os.path.join(base_dir, "data", "genuine", "LKR_5000", '*.jpg'))[:10]
    
    img_data_list = [(p, "LKR_1000") for p in gen_1000] + [(p, "LKR_5000") for p in gen_5000]
    
    if not img_data_list:
        print("No images found.")
        return
        
    print(f"Running stress test on {len(img_data_list)} combined genuine notes (LKR 1000 & 5000)...")
    
    results = []
    
    # Rotational Skew
    for angle in [5, 8, 15]:
        fn_rate = test_condition(img_data_list, apply_rotation, angle)
        status = "Failed (ORB Lost)" if fn_rate >= 80.0 else "Successful"
        results.append({
            "Degradation Stress Factor": "Rotational Skew", 
            "Applied Intensity Level": f"±{angle}°", 
            "Stage 1 Alignment Status": status,
            "False Negative Rate (Flipped to False Positive)": f"{fn_rate:.1f}%"
        })
        print(f"Rotational Skew ±{angle}° -> False Negative: {fn_rate:.1f}%")
        
    # Illumination Shift
    for shift in [10, 15, 30]:
        fn_rate = test_condition(img_data_list, apply_illumination, shift)
        status = "Failed (ORB Lost)" if fn_rate >= 80.0 else "Successful"
        results.append({
            "Degradation Stress Factor": "Illumination Shift", 
            "Applied Intensity Level": f"±{shift} intensity", 
            "Stage 1 Alignment Status": status,
            "False Negative Rate (Flipped to False Positive)": f"{fn_rate:.1f}%"
        })
        print(f"Illumination Shift ±{shift} -> False Negative: {fn_rate:.1f}%")
        
    # Gaussian Noise
    for sigma in [5, 10, 20]:
        fn_rate = test_condition(img_data_list, apply_noise, sigma)
        status = "Failed (ORB Lost)" if fn_rate >= 80.0 else "Successful"
        results.append({
            "Degradation Stress Factor": "Gaussian Noise", 
            "Applied Intensity Level": f"σ = {sigma}", 
            "Stage 1 Alignment Status": status,
            "False Negative Rate (Flipped to False Positive)": f"{fn_rate:.1f}%"
        })
        print(f"Gaussian Noise sigma={sigma} -> False Negative: {fn_rate:.1f}%")
        
    df = pd.DataFrame(results)
    
    out_path = os.path.join(base_dir, 'Stress_Test_Matrix.csv')
    df.to_csv(out_path, index=False)
    print(f"\nStress Test Matrix saved to {out_path}")
    print(df.to_string())

if __name__ == "__main__":
    run_stress_test()
