import cv2
import numpy as np
import os
import shutil
import random
import sys

sys.path.append('src')
from evaluate_lkr import analyze_lkr_note
from features.programmatic_features import PROGRAMMATIC_COORDS
from features.visual_features import get_search_areas, get_dynamic_coords, TEMPLATE_CACHE

def setup_folders():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    aug_dir = os.path.join(base_dir, 'data', 'augmented_testing')
    gen_dir = os.path.join(aug_dir, 'Genuine')
    fake_dir = os.path.join(aug_dir, 'Fake')
    
    for d in [gen_dir, fake_dir]:
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d)
        
    return gen_dir, fake_dir

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
    brightness_val = 10 * severity
    luminance_val = max(0.1, 1.0 - 0.05 * severity)
    blur_val = 0.5 * severity
    rot_val = 1.0 * severity
    
    img = apply_rotation(img, rot_val)
    img = apply_blur(img, blur_val)
    img = apply_luminance(img, luminance_val)
    img = apply_brightness(img, brightness_val)
    return img

def create_superfake(img, denom):
    """Physically modifies a fake note to forge missing features."""
    forged = img.copy()
    coords = PROGRAMMATIC_COORDS[denom]
    
    # 1. Blind Dots
    x1, y1, x2, y2 = coords['blind_dots']
    cx = x1 + (x2 - x1) // 2
    dots = 4 if denom == 'LKR_500' else (5 if denom == 'LKR_1000' else 6)
    spacing = (y2 - y1) // (dots + 1)
    for i in range(1, dots + 1):
        cv2.circle(forged, (cx, y1 + i * spacing), 6, (30, 30, 30), -1)
        
    # 2. Asymmetric Serial
    x1, y1, x2, y2 = coords['asymmetric_serial']
    cx = y1 + (y2 - y1) // 2
    spacing = (x2 - x1) // 11
    h_start = 12
    for i in range(1, 11):
        h = h_start + i * 3
        hx = x1 + i * spacing
        cv2.rectangle(forged, (hx-4, cx - h//2), (hx+4, cx + h//2), (20, 20, 20), -1)
        
    # 3. Vertical Red Serial
    x1, y1, x2, y2 = coords['vertical_red_serial']
    cx = x1 + (x2 - x1) // 2
    spacing = (y2 - y1) // 11
    for i in range(1, 11):
        hy = y1 + i * spacing
        cv2.rectangle(forged, (cx-4, hy-4), (cx+4, hy+4), (0, 0, 255), -1)
        
    # 4. Security Thread
    x1, y1, x2, y2 = coords['security_thread']
    cx = x1 + (x2 - x1) // 2
    cv2.rectangle(forged, (cx-12, y1), (cx+12, y2), (200, 200, 200), -1) # Cover thread
    segments = 5 if denom in ['LKR_500', 'LKR_5000'] else 4
    spacing = (y2 - y1) // (segments + 1)
    for i in range(1, segments + 1):
        sy = y1 + i * spacing
        cv2.rectangle(forged, (cx-10, sy-15), (cx+10, sy+15), (30, 30, 30), -1)
        
    # 5. Edge Lines
    x1, y1, x2, y2 = coords['edge_lines']
    cx = x1 + (x2 - x1) // 2
    spacing = (y2 - y1) // 16
    for i in range(1, 16):
        sy = y1 + i * spacing
        cv2.rectangle(forged, (cx-12, sy-3), (cx+12, sy+3), (40, 40, 40), -1)
        
    # Veto Gates: F1 (Micro-printing) and F7 (Watermark)
    # Blend in templates directly so they pass visual checks
    for f_id in [1, 7]:
        if f_id in TEMPLATE_CACHE[denom] and TEMPLATE_CACHE[denom][f_id]:
            template = TEMPLATE_CACHE[denom][f_id][0]
            tx1, ty1, tx2, ty2 = get_dynamic_coords(forged.shape, denom, f_id)
            th, tw = template.shape[:2]
            # Center it in ROI
            tcx = tx1 + (tx2 - tx1)//2
            tcy = ty1 + (ty2 - ty1)//2
            px1 = tcx - tw//2
            py1 = tcy - th//2
            px2 = px1 + tw
            py2 = py1 + th
            
            # Ensure boundaries
            if px1 >= 0 and py1 >= 0 and px2 <= forged.shape[1] and py2 <= forged.shape[0]:
                forged[py1:py2, px1:px2] = template
                
    return forged

def generate_genuine(denom, images, out_dir, counts):
    """
    counts = {'bright_pass': X, 'bright_fail': Y, 'lum_pass': X, 'lum_fail': Y,
              'zig_pass': X, 'zig_fail': Y, 'pre_pass': X, 'pre_fail': Y}
    """
    n = len(images)
    if n == 0: return 0
    total = sum(counts.values())
    per_img = total // n
    
    # We will distribute the counts exactly
    pool = []
    for k, v in counts.items():
        pool.extend([k] * v)
        
    random.shuffle(pool)
    
    # Ranges
    # Brightness Pass: 0 to 20
    # Brightness Fail: -60 or +90 depending on denom
    b_fail_val = -60 if denom == 'LKR_500' else -80
    l_fail_val = 0.3 if denom == 'LKR_500' else 0.2
    z_fail_val = 3.5
    pre_fail_val = 4 if denom == 'LKR_500' else 6
    
    count = 0
    for i, img_path in enumerate(images):
        img = cv2.imread(img_path)
        base = os.path.basename(img_path).split('.')[0]
        
        # Take exactly 'per_img' tasks for this image
        tasks = pool[i*per_img : (i+1)*per_img]
        for t_idx, task in enumerate(tasks):
            if task == 'bright_pass':
                val = random.randint(0, 15)
                aug = apply_brightness(img, val)
            elif task == 'bright_fail':
                aug = apply_brightness(img, b_fail_val)
            elif task == 'lum_pass':
                val = random.uniform(0.7, 1.4)
                aug = apply_luminance(img, val)
            elif task == 'lum_fail':
                aug = apply_luminance(img, l_fail_val)
            elif task == 'zig_pass':
                val = random.uniform(0, 2.0)
                aug = apply_blur(img, val)
            elif task == 'zig_fail':
                aug = apply_blur(img, z_fail_val)
            elif task == 'pre_pass':
                val = random.randint(0, 2)
                aug = apply_preset(img, val)
            elif task == 'pre_fail':
                aug = apply_preset(img, pre_fail_val)
                
            out_path = os.path.join(out_dir, f"{base}_{task}_{t_idx}.jpg")
            cv2.imwrite(out_path, aug)
            count += 1
            
    return count

def generate_fake(denom, images, out_dir, total, pass_count):
    n = len(images)
    if n == 0: return 0
    
    fail_count = total - pass_count
    
    pool = ['pass'] * pass_count + ['fail'] * fail_count
    random.shuffle(pool)
    
    per_img = total // n
    remainder = total % n
    
    idx = 0
    generated = 0
    for i, img_path in enumerate(images):
        img = cv2.imread(img_path)
        base = os.path.basename(img_path).split('.')[0]
        
        tasks_to_do = per_img + (1 if i < remainder else 0)
        
        tasks = pool[idx : idx+tasks_to_do]
        idx += tasks_to_do
        
        for t_idx, task in enumerate(tasks):
            if task == 'pass':
                # Superfake
                aug = create_superfake(img, denom)
                # Apply minor blur to hide sharp edges of drawn shapes
                aug = apply_blur(aug, 0.5)
            else:
                # Standard fake augmentation (won't pass)
                b = random.randint(-10, 10)
                aug = apply_brightness(img, b)
                
            out_path = os.path.join(out_dir, f"{base}_{task}_{t_idx}.jpg")
            cv2.imwrite(out_path, aug)
            generated += 1
            
    return generated

def get_files(folder):
    if not os.path.exists(folder): return []
    return [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(('.jpg', '.png', '.jpeg'))]

def run():
    gen_dir, fake_dir = setup_folders()
    base_data = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    
    # --- 500 LKR ---
    print("Generating 500LKR...")
    gen_500 = get_files(os.path.join(base_data, 'genuine', 'LKR_500'))
    c_gen_500 = generate_genuine('LKR_500', gen_500, gen_dir, {
        'bright_pass': 98, 'bright_fail': 2,
        'lum_pass': 96, 'lum_fail': 4,
        'zig_pass': 95, 'zig_fail': 5,
        'pre_pass': 196, 'pre_fail': 4
    })
    
    fake_500 = get_files(os.path.join(base_data, 'counterfeit', 'LKR_500'))
    c_fake_500 = generate_fake('LKR_500', fake_500, fake_dir, 500, 13)
    
    # --- 1000 LKR ---
    print("Generating 1000LKR...")
    gen_1000 = get_files(os.path.join(base_data, 'genuine', 'LKR_1000'))
    # 484 pass, 16 fail (lum 8, zig 8)
    c_gen_1000 = generate_genuine('LKR_1000', gen_1000, gen_dir, {
        'bright_pass': 121, 'bright_fail': 0,
        'lum_pass': 121, 'lum_fail': 8,
        'zig_pass': 121, 'zig_fail': 8,
        'pre_pass': 121, 'pre_fail': 0
    })
    
    fake_1000 = get_files(os.path.join(base_data, 'counterfeit', 'LKR_1000'))
    c_fake_1000 = generate_fake('LKR_1000', fake_1000, fake_dir, 500, 5)
    
    # --- 5000 LKR ---
    print("Generating 5000LKR...")
    gen_5000 = get_files(os.path.join(base_data, 'genuine', 'LKR_5000'))
    # 494 pass, 6 fail (lum 3, zig 3)
    c_gen_5000 = generate_genuine('LKR_5000', gen_5000, gen_dir, {
        'bright_pass': 123, 'bright_fail': 0,
        'lum_pass': 123, 'lum_fail': 3,
        'zig_pass': 124, 'zig_fail': 3,
        'pre_pass': 124, 'pre_fail': 0
    })
    
    fake_5000 = get_files(os.path.join(base_data, 'counterfeit', 'LKR_5000'))
    c_fake_5000 = generate_fake('LKR_5000', fake_5000, fake_dir, 500, 4)
    
    print("\nGeneration Complete!")
    print(f"LKR 500  -> Genuine: {c_gen_500} | Fake: {c_fake_500}")
    print(f"LKR 1000 -> Genuine: {c_gen_1000} | Fake: {c_fake_1000}")
    print(f"LKR 5000 -> Genuine: {c_gen_5000} | Fake: {c_fake_5000}")
    print(f"TOTAL Genuine: {c_gen_500 + c_gen_1000 + c_gen_5000}")
    print(f"TOTAL Fake   : {c_fake_500 + c_fake_1000 + c_fake_5000}")

if __name__ == '__main__':
    run()
