import cv2
import numpy as np
import os

# =====================================================================
# DYNAMIC PERCENTAGE-BASED ANCHORS
# =====================================================================
LKR_1000_SEARCH_AREAS_PCT = {
    1: (0.008, 0.710, 0.149, 0.988),
    2: (0.012, 0.534, 0.220, 0.988),
    3: (0.162, 0.638, 0.340, 0.948),
    4: (0.760, 0.200, 0.980, 0.750),
    5: (0.820, 0.020, 0.930, 0.250),
    6: (0.598, 0.788, 0.818, 0.982),
    7: (0.168, 0.318, 0.279, 0.758)
}

LKR_5000_SEARCH_AREAS_PCT = {
    1: (0.004, 0.710, 0.138, 0.990),
    2: (0.007, 0.556, 0.195, 0.978),
    3: (0.138, 0.632, 0.310, 0.960),
    4: (0.755, 0.160, 0.991, 0.814),
    5: (0.812, 0.008, 0.941, 0.278),
    6: (0.601, 0.794, 0.810, 0.984),
    7: (0.167, 0.308, 0.269, 0.754)
}

LKR_500_SEARCH_AREAS_PCT = {
    1: (0.010, 0.760, 0.120, 0.960),
    2: (0.015, 0.590, 0.210, 0.960),
    3: (0.150, 0.680, 0.310, 0.930),
    4: (0.770, 0.200, 0.980, 0.780),
    5: (0.820, 0.040, 0.950, 0.240),
    6: (0.610, 0.830, 0.830, 0.960),
    7: (0.160, 0.350, 0.260, 0.720)
}

# =====================================================================
# TEMPLATE CACHE
# =====================================================================
TEMPLATE_CACHE = {'LKR_500': {}, 'LKR_1000': {}, 'LKR_5000': {}}
TEMPLATE_ORB_CACHE = {'LKR_500': {}, 'LKR_1000': {}, 'LKR_5000': {}}

GLOBAL_ORB = cv2.ORB_create(500)
GLOBAL_BF = cv2.BFMatcher(cv2.NORM_HAMMING)

def load_templates():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    for denom in ['LKR_500', 'LKR_1000', 'LKR_5000']:
        denom_dir = os.path.join(base_dir, 'data', 'templates', denom)
        if not os.path.exists(denom_dir): continue
        
        for feature_id in range(1, 8):
            feature_dir = os.path.join(denom_dir, f'Feature {feature_id}')
            if not os.path.exists(feature_dir): continue
            
            TEMPLATE_CACHE[denom][feature_id] = []
            TEMPLATE_ORB_CACHE[denom][feature_id] = []
            for i in range(1, 11):
                img_path = os.path.join(feature_dir, f'{i}.jpg')
                img = cv2.imread(img_path)
                if img is not None:
                    TEMPLATE_CACHE[denom][feature_id].append(img)
                    _, des = GLOBAL_ORB.detectAndCompute(img, None)
                    TEMPLATE_ORB_CACHE[denom][feature_id].append(des)

load_templates()

# =====================================================================
# ALGORITHM 1: VISUAL FEATURES (Context-Aware Architecture)
# =====================================================================

def calculate_ssim(img1, img2):
    """Calculates SSIM with CLAHE pre-processing to eliminate shadow-induced False Negatives."""
    from skimage.metrics import structural_similarity
    
    img2_resized = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) if len(img1.shape) == 3 else img1
    gray2 = cv2.cvtColor(img2_resized, cv2.COLOR_BGR2GRAY) if len(img2_resized.shape) == 3 else img2_resized
    
    score, _ = structural_similarity(gray1, gray2, full=True)
    return score

def get_search_areas(denom):
    if denom == 'LKR_500':
        return LKR_500_SEARCH_AREAS_PCT
    elif denom == 'LKR_1000':
        return LKR_1000_SEARCH_AREAS_PCT
    elif denom == 'LKR_5000':
        return LKR_5000_SEARCH_AREAS_PCT
    return LKR_1000_SEARCH_AREAS_PCT

def get_dynamic_coords(img_shape, denom, feature_id):
    pct_map = get_search_areas(denom)
    px1, py1, px2, py2 = pct_map[feature_id]
    h, w = img_shape[:2]
    
    x1, y1 = int(px1 * w), int(py1 * h)
    x2, y2 = int(px2 * w), int(py2 * h)
    return x1, y1, x2, y2

def verify_visual_feature(img, denom, feature_id):
    if feature_id not in TEMPLATE_CACHE[denom] or not TEMPLATE_CACHE[denom][feature_id]:
        return False, "Templates not found", None, 0.0

    x1, y1, x2, y2 = get_dynamic_coords(img.shape, denom, feature_id)
    
    pad = 30
    img_h, img_w = img.shape[:2]
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(img_w, x2 + pad)
    y2 = min(img_h, y2 + pad)
    
    search_img = img[y1:y2, x1:x2]
    
    max_ssim = -1.0
    best_crop = None
    best_corr = -1.0
    max_orb_matches = 0
    max_color_corr = -1.0
    
    kp_search, des_search = GLOBAL_ORB.detectAndCompute(search_img, None)
    
    for i, template in enumerate(TEMPLATE_CACHE[denom][feature_id]):
        # Preprocessing block to eliminate shadow-induced False Negatives
        if feature_id in [1, 4, 5, 7]:
            if feature_id == 7:
                # Watermark (F7) requires stronger CLAHE + Gaussian blur for faint printing
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
                search_proc = cv2.GaussianBlur(clahe.apply(cv2.cvtColor(search_img, cv2.COLOR_BGR2GRAY)), (5, 5), 0)
                template_proc = cv2.GaussianBlur(clahe.apply(cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)), (5, 5), 0)
            else:
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                search_proc = clahe.apply(cv2.cvtColor(search_img, cv2.COLOR_BGR2GRAY))
                template_proc = clahe.apply(cv2.cvtColor(template, cv2.COLOR_BGR2GRAY))
            
            # Apply Gaussian Blur for F4 to suppress halftone printing noise
            if feature_id == 4:
                search_proc = cv2.GaussianBlur(search_proc, (5, 5), 0)
                template_proc = cv2.GaussianBlur(template_proc, (5, 5), 0)
                
            # Improve Micro-Printing (F1) for LKR_1000 and LKR_500 using Median Blur
            # LKR_500 is equally sensitive to Zigma noise — same treatment applied.
            if feature_id == 1 and denom in ['LKR_1000', 'LKR_500']:
                search_proc = cv2.medianBlur(search_proc, 3)
                template_proc = cv2.medianBlur(template_proc, 3)
                
            res = cv2.matchTemplate(search_proc, template_proc, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(res)
            
            tx, ty = max_loc
            th, tw = template.shape[:2]
            extracted_feature = search_img[ty:ty+th, tx:tx+tw]
            
            # Recalculate SSIM on the extracted region using the exact same processed versions
            ext_gray = search_proc[ty:ty+th, tx:tx+tw]
            tmp_gray = template_proc
                
            if feature_id == 7:
                # Calculate SSIM directly on grayscale for F7 due to blur_clahe
                score = calculate_ssim(ext_gray, tmp_gray)
            else:
                ext_bgr = cv2.cvtColor(ext_gray, cv2.COLOR_GRAY2BGR)
                tmp_bgr = cv2.cvtColor(tmp_gray, cv2.COLOR_GRAY2BGR)
                
                # Apply Gaussian blur to suppress noise amplified by CLAHE
                ext_blur = cv2.GaussianBlur(ext_bgr, (5, 5), 0)
                tmp_blur = cv2.GaussianBlur(tmp_bgr, (5, 5), 0)
                score = calculate_ssim(ext_blur, tmp_blur)
        else:
            # Standard fast BGR template matching
            res = cv2.matchTemplate(search_img, template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(res)
            
            tx, ty = max_loc
            th, tw = template.shape[:2]
            extracted_feature = search_img[ty:ty+th, tx:tx+tw]
            
            # Apply Gaussian blur for F3 and F6 SSIM to suppress halftone dots
            if feature_id in [3, 6]:
                if feature_id == 6 and denom == 'LKR_500':
                    # For LKR_500 motif (F6), apply CLAHE normalization first.
                    # This removes sensitivity to brightness/luminance augmentations
                    # so that the SSIM score reflects structural similarity only.
                    clahe_f6 = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                    gray_ext = clahe_f6.apply(cv2.cvtColor(extracted_feature, cv2.COLOR_BGR2GRAY))
                    gray_tmp = clahe_f6.apply(cv2.cvtColor(template, cv2.COLOR_BGR2GRAY))
                    ext_blur = cv2.GaussianBlur(cv2.cvtColor(gray_ext, cv2.COLOR_GRAY2BGR), (5, 5), 0)
                    tmp_blur = cv2.GaussianBlur(cv2.cvtColor(gray_tmp, cv2.COLOR_GRAY2BGR), (5, 5), 0)
                else:
                    ext_blur = cv2.GaussianBlur(extracted_feature, (5, 5), 0)
                    tmp_blur = cv2.GaussianBlur(template, (5, 5), 0)
                score = calculate_ssim(ext_blur, tmp_blur)
            else:
                score = calculate_ssim(extracted_feature, template)
        
        # Color Histogram logic for motifs to prevent Hue-shifted fakes from passing
        c_corr = 0.0
        if feature_id in [2, 3, 5, 6]:
            hsv_ext = cv2.cvtColor(extracted_feature, cv2.COLOR_BGR2HSV)
            hsv_tmp = cv2.cvtColor(template, cv2.COLOR_BGR2HSV)
            hist_ext = cv2.calcHist([hsv_ext], [0, 1], None, [50, 50], [0, 180, 0, 256])
            hist_tmp = cv2.calcHist([hsv_tmp], [0, 1], None, [50, 50], [0, 180, 0, 256])
            cv2.normalize(hist_ext, hist_ext, 0, 1, cv2.NORM_MINMAX)
            cv2.normalize(hist_tmp, hist_tmp, 0, 1, cv2.NORM_MINMAX)
            c_corr = cv2.compareHist(hist_tmp, hist_ext, cv2.HISTCMP_CORREL)
            if c_corr > max_color_corr: max_color_corr = c_corr
        
        if score > max_ssim:
            max_ssim = score
            best_crop = extracted_feature.copy()
            best_corr = max_val
            
        des_temp = TEMPLATE_ORB_CACHE[denom][feature_id][i]
        if des_search is not None and des_temp is not None and len(des_search) > 5 and len(des_temp) > 5:
            try:
                matches = GLOBAL_BF.knnMatch(des_temp, des_search, k=2)
                good = []
                for m_n in matches:
                    if len(m_n) == 2:
                        m, n = m_n
                        if m.distance < 0.75 * n.distance:
                            good.append(m)
                if len(good) > max_orb_matches:
                    max_orb_matches = len(good)
            except cv2.error:
                pass
            
    # Determine Thresholds by Denomination and Feature
    thresh = {
        'LKR_500': {
            1: {'ssim': 0.65, 'tm': 0.70},  # Reduced from 0.70
            2: {'orb': 30, 'color': 0.70},
            3: {'ssim': 0.70, 'tm': 0.75},  # Reduced from 0.75
            4: {'ssim': 0.65, 'tm': 0.70},  # Reduced from 0.70
            5: {'ssim': 0.70, 'tm': 0.75},  # Reduced from 0.75
            6: {'ssim': 0.70, 'tm': 0.70},  # Reduced from 0.75
            7: {'ssim': 0.50, 'tm': 0.50},
        },
        'LKR_1000': {
            1: {'ssim': 0.50, 'tm': 0.70},
            2: {'orb': 35, 'color': 0.75},
            3: {'ssim': 0.60, 'tm': 0.45},
            4: {'ssim': 0.60, 'tm': 0.75},
            5: {'ssim': 0.70, 'tm': 0.45},
            6: {'ssim': 0.75, 'tm': 0.65},
            7: {'ssim': 0.33, 'tm': 0.53},
        },
        'LKR_5000': {
            1: {'ssim': 0.70, 'tm': 0.45},
            2: {'orb': 15, 'color': 0.75},
            3: {'ssim': 0.75, 'tm': 0.55},
            4: {'ssim': 0.60, 'tm': 0.70},
            5: {'ssim': 0.70, 'tm': 0.50},
            6: {'ssim': 0.80, 'tm': 0.70},
            7: {'ssim': 0.59, 'tm': 0.40},
        }
    }
    
    t = thresh.get(denom, thresh['LKR_1000']).get(feature_id, {'ssim': 0.60, 'tm': 0.75})
    
    if feature_id == 2:
        req_orb = t.get('orb', 12)
        req_color = t.get('color', 0.5)
        passed = (max_orb_matches > req_orb) and (max_color_corr > req_color)
        if (max_ssim > 0.40 and best_corr > 0.75) and (max_color_corr > req_color):
            passed = True
        msg = f"ORB: {max_orb_matches} | Color: {max_color_corr:.2f}"
    else:
        req_ssim = t.get('ssim', 0.60)
        req_tm = t.get('tm', 0.75)
        passed = (max_ssim > req_ssim) and (best_corr > req_tm)
        msg = f"SSIM: {max_ssim:.3f} | TM: {best_corr:.3f}"
        
    return passed, msg, best_crop, max_ssim
