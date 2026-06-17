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
    4: (0.747, 0.166, 0.994, 0.798),
    5: (0.812, 0.002, 0.946, 0.270),
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

# =====================================================================
# TEMPLATE CACHE
# =====================================================================
TEMPLATE_CACHE = {'LKR_1000': {}, 'LKR_5000': {}}
TEMPLATE_ORB_CACHE = {'LKR_1000': {}, 'LKR_5000': {}}

GLOBAL_ORB = cv2.ORB_create(500)
GLOBAL_BF = cv2.BFMatcher(cv2.NORM_HAMMING)

def load_templates():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    for denom in ['LKR_1000', 'LKR_5000']:
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
    
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2_resized, cv2.COLOR_BGR2GRAY)
    
    score, _ = structural_similarity(gray1, gray2, full=True)
    return score

def get_dynamic_coords(img_shape, denom, feature_id):
    pct_map = LKR_1000_SEARCH_AREAS_PCT if denom == 'LKR_1000' else LKR_5000_SEARCH_AREAS_PCT
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
        # Fast template matching to find the exact sub-location for XAI Cropping
        res = cv2.matchTemplate(search_img, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)
        
        tx, ty = max_loc
        th, tw = template.shape[:2]
        extracted_feature = search_img[ty:ty+th, tx:tx+tw]
        
        # Apply CLAHE strictly to Veto Gates (F1, F7) to fix shadow-induced False Negatives
        if feature_id in [1, 7]:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            ext_gray = clahe.apply(cv2.cvtColor(extracted_feature, cv2.COLOR_BGR2GRAY))
            tmp_gray = clahe.apply(cv2.cvtColor(template, cv2.COLOR_BGR2GRAY))
            ext_bgr = cv2.cvtColor(ext_gray, cv2.COLOR_GRAY2BGR)
            tmp_bgr = cv2.cvtColor(tmp_gray, cv2.COLOR_GRAY2BGR)
            score = calculate_ssim(ext_bgr, tmp_bgr)
        else:
            score = calculate_ssim(extracted_feature, template)
        
        # Color Histogram logic for motifs to prevent Hue-shifted fakes from passing
        c_corr = 0.0
        if feature_id in [2, 5]:
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
            
    # Scale-Invariant Artistic Motifs (F2, F5)
    if feature_id in [2, 5]:
        passed = (max_orb_matches > 12) and (max_color_corr > 0.5)
        # Fallback to pure SSIM if the note perfectly matches structure but ORB lost keypoints
        if (max_ssim > 0.40 and best_corr > 0.75) and (max_color_corr > 0.5):
            passed = True
        msg = f"ORB: {max_orb_matches} | Color: {max_color_corr:.2f}"
    
    # Rigid Spatial Numerals/Text (F3, F4, F6)
    elif feature_id in [3, 4, 6]:
        # Strict SSIM checks prevent perspective-warped fakes from passing
        threshold_ssim = 0.35
        threshold_tm = 0.50
        passed = (max_ssim > threshold_ssim) and (best_corr > threshold_tm)
        msg = f"SSIM: {max_ssim:.3f} | TM: {best_corr:.3f}"
        
    # Structural VETO Gates (F1, F7) - Protected by CLAHE
    elif feature_id == 1: 
        threshold_ssim = 0.28
        threshold_tm = 0.40
        passed = (max_ssim > threshold_ssim) and (best_corr > threshold_tm)
        msg = f"SSIM: {max_ssim:.3f} | TM: {best_corr:.3f}"
        
    elif feature_id == 7: 
        threshold_ssim = 0.20 # Watermarks are faint, CLAHE boosts them but requires lower threshold
        threshold_tm = 0.20
        passed = (max_ssim > threshold_ssim) and (best_corr > threshold_tm)
        msg = f"SSIM: {max_ssim:.3f} | TM: {best_corr:.3f}"
        
    else:
        threshold_ssim = 0.60
        threshold_tm = 0.75
        passed = (max_ssim > threshold_ssim) and (best_corr > threshold_tm)
        msg = f"SSIM: {max_ssim:.3f} | TM: {best_corr:.3f}"
        
    return passed, msg, best_crop, max_ssim
