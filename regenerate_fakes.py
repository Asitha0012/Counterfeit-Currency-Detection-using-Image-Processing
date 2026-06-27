import cv2
import numpy as np
import os
import shutil
import random
import glob

def setup_folders():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    fake_dir = os.path.join(base_dir, 'data', 'augmented_testing', 'Fake')
    
    if os.path.exists(fake_dir):
        shutil.rmtree(fake_dir)
    os.makedirs(fake_dir)
    
    for denom in ['LKR_500', 'LKR_1000', 'LKR_5000']:
        os.makedirs(os.path.join(fake_dir, denom))
        
    return fake_dir

def apply_brightness(img, beta):
    return cv2.convertScaleAbs(img, alpha=1, beta=beta)

def apply_luminance(img, alpha):
    return cv2.convertScaleAbs(img, alpha=alpha, beta=0)

def apply_blur(img, sigma):
    if sigma <= 0: return img
    return cv2.GaussianBlur(img, (0, 0), sigmaX=sigma)

def apply_hsv(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    
    # Random shifts
    h_shift = random.randint(-15, 15)
    s_shift = random.randint(-40, 40)
    v_shift = random.randint(-30, 30)
    
    h = np.clip(h.astype(np.int16) + h_shift, 0, 179).astype(np.uint8)
    s = np.clip(s.astype(np.int16) + s_shift, 0, 255).astype(np.uint8)
    v = np.clip(v.astype(np.int16) + v_shift, 0, 255).astype(np.uint8)
    
    hsv_new = cv2.merge((h, s, v))
    return cv2.cvtColor(hsv_new, cv2.COLOR_HSV2BGR)

def get_random_from_range(range_tuple):
    # ranges can be negative, so we sort them
    r = sorted(range_tuple)
    if isinstance(r[0], int) and isinstance(r[1], int):
        return random.randint(r[0], r[1])
    else:
        return random.uniform(r[0], r[1])

def generate_for_denom(denom, source_images, out_dir, params):
    total_needed = 500
    n = len(source_images)
    
    # We will define how many of each augmentation type to do
    # HSV: 25, Brightness: 158, Luminance: 158, Zigma: 159
    tasks = []
    
    # HSV
    tasks.extend(['hsv'] * 25)
    
    # Brightness (80% pass, 20% fail)
    tasks.extend(['bright_pass'] * 126)
    tasks.extend(['bright_fail'] * 32)
    
    # Luminance (80% pass, 20% fail)
    tasks.extend(['lum_pass'] * 126)
    tasks.extend(['lum_fail'] * 32)
    
    # Zigma (80% pass, 20% fail)
    tasks.extend(['zig_pass'] * 127)
    tasks.extend(['zig_fail'] * 32)
    
    random.shuffle(tasks)
    
    idx = 0
    generated = 0
    
    per_img = total_needed // n
    remainder = total_needed % n
    
    for i, img_path in enumerate(source_images):
        img = cv2.imread(img_path)
        base = os.path.basename(img_path).split('.')[0]
        
        tasks_to_do = per_img + (1 if i < remainder else 0)
        
        img_tasks = tasks[idx : idx + tasks_to_do]
        idx += tasks_to_do
        
        for t_idx, task in enumerate(img_tasks):
            if task == 'hsv':
                aug = apply_hsv(img)
            elif task == 'bright_pass':
                val = get_random_from_range(params['bright_pass'])
                aug = apply_brightness(img, val)
            elif task == 'bright_fail':
                val = get_random_from_range(random.choice(params['bright_fail']))
                aug = apply_brightness(img, val)
            elif task == 'lum_pass':
                val = get_random_from_range(params['lum_pass'])
                aug = apply_luminance(img, val)
            elif task == 'lum_fail':
                val = get_random_from_range(random.choice(params['lum_fail']))
                aug = apply_luminance(img, val)
            elif task == 'zig_pass':
                val = get_random_from_range(params['zig_pass'])
                aug = apply_blur(img, val)
            elif task == 'zig_fail':
                val = get_random_from_range(params['zig_fail'])
                aug = apply_blur(img, val)
                
            out_path = os.path.join(out_dir, f"{base}_{task}_{t_idx}.jpg")
            cv2.imwrite(out_path, aug)
            generated += 1
            
    return generated

def run():
    fake_dir = setup_folders()
    base_data = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'counterfeit')
    
    print("Generating Fakes...")
    
    # --- 500 LKR (8 notes) ---
    fake_500 = glob.glob(os.path.join(base_data, 'LKR_500', '*.jpg'))[:8]
    p_500 = {
        'bright_pass': (-40, 20), 'bright_fail': [(-60, -41), (21, 40)],
        'lum_pass': (0.5, 1.6), 'lum_fail': [(0.3, 0.49), (1.61, 1.8)],
        'zig_pass': (0, 2.5), 'zig_fail': (2.6, 3.5)
    }
    g_500 = generate_for_denom('LKR_500', fake_500, os.path.join(fake_dir, 'LKR_500'), p_500)
    print(f"Generated {g_500} fake notes for LKR_500")
    
    # --- 1000 LKR (5 notes, exclude copy) ---
    fake_1000_all = glob.glob(os.path.join(base_data, 'LKR_1000', '*.jpg'))
    fake_1000 = [f for f in fake_1000_all if 'copy' not in f.lower()][:5]
    p_1000 = {
        'bright_pass': (-50, 70), 'bright_fail': [(-70, -51), (71, 90)],
        'lum_pass': (0.4, 1.5), 'lum_fail': [(0.2, 0.39), (1.51, 1.7)],
        'zig_pass': (0, 2.5), 'zig_fail': (2.6, 3.5)
    }
    g_1000 = generate_for_denom('LKR_1000', fake_1000, os.path.join(fake_dir, 'LKR_1000'), p_1000)
    print(f"Generated {g_1000} fake notes for LKR_1000")
    
    # --- 5000 LKR (6 notes) ---
    fake_5000 = glob.glob(os.path.join(base_data, 'LKR_5000', '*.jpg'))[:6]
    p_5000 = {
        'bright_pass': (-60, 100), 'bright_fail': [(-80, -61), (101, 120)],
        'lum_pass': (0.3, 1.9), 'lum_fail': [(0.1, 0.29), (1.91, 2.2)],
        'zig_pass': (0, 2.5), 'zig_fail': (2.6, 3.5)
    }
    g_5000 = generate_for_denom('LKR_5000', fake_5000, os.path.join(fake_dir, 'LKR_5000'), p_5000)
    print(f"Generated {g_5000} fake notes for LKR_5000")

if __name__ == '__main__':
    run()
