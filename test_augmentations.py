import cv2
import numpy as np
import os
import sys

sys.path.append('src')
from evaluate_lkr import analyze_lkr_note

def apply_brightness(img, beta):
    return cv2.convertScaleAbs(img, alpha=1, beta=beta)

def apply_luminance(img, alpha):
    return cv2.convertScaleAbs(img, alpha=alpha, beta=0)

def apply_blur(img, sigma):
    if sigma <= 0: return img
    return cv2.GaussianBlur(img, (0, 0), sigmaX=sigma)

def apply_rotation(img, angle):
    if angle == 0: return img
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), borderValue=(255,255,255))

def apply_preset(img, severity):
    # severity is a multiplier for distortion
    brightness_val = 10 * severity
    luminance_val = max(0.1, 1.0 - 0.05 * severity) # darkening or lowering contrast
    blur_val = 0.5 * severity
    rot_val = 1.0 * severity
    
    img = apply_rotation(img, rot_val)
    img = apply_blur(img, blur_val)
    img = apply_luminance(img, luminance_val)
    img = apply_brightness(img, brightness_val)
    return img

def test_image(img, denom, temp_path):
    cv2.imwrite(temp_path, img)
    # redirect stdout to hide prints from evaluate_lkr if any
    old_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')
    flat_verdict, robust_verdict, message, final_score, feature_statuses, feature_images = analyze_lkr_note(temp_path, denom)
    sys.stdout = old_stdout
    return robust_verdict

def find_failure_point(img, denom, temp_path, apply_func, start_val, step, max_iters=50):
    val = start_val
    passed_vals = []
    failed_val = None
    
    for _ in range(max_iters):
        augmented = apply_func(img, val)
        passed = test_image(augmented, denom, temp_path)
        if passed:
            passed_vals.append(val)
        else:
            failed_val = val
            break
        val += step
        
    return passed_vals, failed_val

def run_tests():
    notes = {
        'LKR_500': 'data/genuine/LKR_500/500_G1.jpg',
        'LKR_1000': 'data/genuine/LKR_1000/1000_G1.jpg',
        'LKR_5000': 'data/genuine/LKR_5000/5000_G1.jpg'
    }
    
    temp_path = "temp_test_aug.jpg"
    
    with open('augmentation_results.txt', 'w') as f:
        for denom, path in notes.items():
            f.write(f"=== Results for {denom} ===\n")
            img = cv2.imread(path)
            if img is None:
                f.write(f"Could not load image {path}\n")
                continue
            
            # Check baseline
            baseline_pass = test_image(img, denom, temp_path)
            if not baseline_pass:
                f.write(f"Baseline (no augmentation) FAILED for {denom}. Cannot proceed with augmentations properly.\n")
                continue
            
            # 1. Brightness
            f.write("\n-- Brightness --\n")
            # Positive
            p_pass, p_fail = find_failure_point(img, denom, temp_path, apply_brightness, 0, 10)
            f.write(f"Positive brightness passed range: 0 to {p_pass[-1] if p_pass else 0}\n")
            f.write(f"Positive brightness failed at: {p_fail}\n")
            # Negative
            n_pass, n_fail = find_failure_point(img, denom, temp_path, apply_brightness, -10, -10)
            f.write(f"Negative brightness passed range: 0 to {n_pass[-1] if n_pass else 0}\n")
            f.write(f"Negative brightness failed at: {n_fail}\n")
            
            # 2. Luminance (Contrast)
            f.write("\n-- Luminance (Contrast multiplier) --\n")
            # Positive ( > 1)
            p_pass, p_fail = find_failure_point(img, denom, temp_path, apply_luminance, 1.0, 0.1)
            f.write(f"Increased luminance passed range: 1.0 to {p_pass[-1]:.1f}\n")
            f.write(f"Increased luminance failed at: {p_fail:.1f} (approx)\n")
            # Negative ( < 1)
            n_pass, n_fail = find_failure_point(img, denom, temp_path, apply_luminance, 0.9, -0.1)
            if n_pass:
                f.write(f"Decreased luminance passed range: 1.0 to {n_pass[-1]:.1f}\n")
            else:
                f.write(f"Decreased luminance passed range: 1.0 to 1.0\n")
            f.write(f"Decreased luminance failed at: {n_fail:.1f} (approx)\n")
            
            # 3. Zigma (Blur)
            f.write("\n-- Zigma (Gaussian Blur Sigma) --\n")
            p_pass, p_fail = find_failure_point(img, denom, temp_path, apply_blur, 0, 0.5)
            f.write(f"Blur passed range: 0 to {p_pass[-1]}\n")
            f.write(f"Blur failed at sigma: {p_fail}\n")
            
            # 4. Rotation
            f.write("\n-- Rotation --\n")
            # Positive
            p_pass, p_fail = find_failure_point(img, denom, temp_path, apply_rotation, 0, 1)
            f.write(f"Positive rotation passed range: 0 to {p_pass[-1]}\n")
            f.write(f"Positive rotation failed at: {p_fail} degrees\n")
            # Negative
            n_pass, n_fail = find_failure_point(img, denom, temp_path, apply_rotation, -1, -1)
            if n_pass:
                f.write(f"Negative rotation passed range: 0 to {n_pass[-1]}\n")
            else:
                f.write(f"Negative rotation passed range: 0 to 0\n")
            f.write(f"Negative rotation failed at: {n_fail} degrees\n")
            
            # 5. Combined Preset
            f.write("\n-- Combined Preset (Increasing Severity) --\n")
            # severity = 0 is baseline
            p_pass, p_fail = find_failure_point(img, denom, temp_path, apply_preset, 0, 1)
            f.write(f"Preset passed severity range: 0 to {p_pass[-1]}\n")
            f.write(f"Preset failed at severity: {p_fail}\n")
            f.write("Preset parameters at failure:\n")
            if p_fail is not None:
                f.write(f"  Brightness: +{10 * p_fail}\n")
                f.write(f"  Luminance: {max(0.1, 1.0 - 0.05 * p_fail):.2f}x\n")
                f.write(f"  Zigma (Blur): {0.5 * p_fail}\n")
                f.write(f"  Rotation: {1.0 * p_fail} degrees\n")
            
            f.write("\n" + "="*40 + "\n\n")

if __name__ == "__main__":
    run_tests()
    if os.path.exists("temp_test_aug.jpg"):
        os.remove("temp_test_aug.jpg")
    print("Done. Results saved to augmentation_results.txt")
