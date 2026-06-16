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
    return cv2.warpAffine(img, M, (w, h), borderValue=(255, 255, 255))

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

def test_condition(img_paths, denom, condition_func, param):
    failed = 0
    total = len(img_paths)
    for img_path in img_paths:
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
    genuine_dir = os.path.join(base_dir, "data", "augmented_testing", "Genuine", "LKR_1000")
    # Take 20 clean genuine notes for the stress test
    img_paths = glob.glob(os.path.join(genuine_dir, '*.jpg'))[:20]
    
    if not img_paths:
        print("No images found.")
        return
        
    print(f"Running stress test on {len(img_paths)} genuine LKR_1000 notes...")
    
    results = []
    
    # Rotational Skew
    for angle in [5, 8, 15]:
        fn_rate = test_condition(img_paths, "LKR_1000", apply_rotation, angle)
        results.append({"Factor": "Rotational Skew", "Intensity": f"±{angle}°", "FN Rate": f"{fn_rate:.1f}%"})
        print(f"Rotational Skew ±{angle}° -> False Negative: {fn_rate:.1f}%")
        
    # Illumination Shift
    for shift in [10, 15, 30]:
        fn_rate = test_condition(img_paths, "LKR_1000", apply_illumination, shift)
        results.append({"Factor": "Illumination Shift", "Intensity": f"±{shift}", "FN Rate": f"{fn_rate:.1f}%"})
        print(f"Illumination Shift ±{shift} -> False Negative: {fn_rate:.1f}%")
        
    # Gaussian Noise
    for sigma in [5, 10, 20]:
        fn_rate = test_condition(img_paths, "LKR_1000", apply_noise, sigma)
        results.append({"Factor": "Gaussian Noise", "Intensity": f"sigma={sigma}", "FN Rate": f"{fn_rate:.1f}%"})
        print(f"Gaussian Noise sigma={sigma} -> False Negative: {fn_rate:.1f}%")
        
    df = pd.DataFrame(results)
    
    brain_dir = os.path.join(os.environ['USERPROFILE'], '.gemini', 'antigravity-ide', 'brain', '8912d3ae-1929-438e-812f-4925e7c66e2a')
    out_path = os.path.join(brain_dir, 'Stress_Test_Matrix.csv')
    df.to_csv(out_path, index=False)
    print(f"\nStress Test Matrix saved to {out_path}")
    print(df.to_string())

if __name__ == "__main__":
    run_stress_test()
