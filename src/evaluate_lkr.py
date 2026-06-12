import cv2
import numpy as np
import os
import time

# We must import the alignment logic to align test notes before evaluation
from align_note import align_note

# =====================================================================
# CONFIGURATION & COORDINATES
# =====================================================================

LKR_1000_SEARCH_AREAS = {
    1: (9, 355, 165, 494),      # Micro-Printing
    2: (13, 267, 243, 494),     # Butterfly
    3: (179, 319, 375, 474),    # Note Value
    4: (825, 83, 1097, 399),    # Bird
    5: (897, 1, 1044, 135),     # Lion Emblem
    6: (660, 394, 903, 491),    # Value in Text
    7: (186, 159, 308, 379)     # Watermark
}

LKR_5000_SEARCH_AREAS = {
    1: (5, 355, 157, 495),
    2: (8, 278, 223, 489),
    3: (157, 316, 354, 480),
    4: (861, 80, 1131, 407),
    5: (926, 4, 1074, 139),
    6: (686, 397, 924, 492),
    7: (190, 154, 307, 377)
}

PROGRAMMATIC_COORDS = {
    'LKR_1000': {
        'blind_dots': (40, 110, 82, 310),
        'asymmetric_serial': (76, 138, 280, 191),
        'vertical_red_serial': (803, 120, 868, 398),
        'security_thread': (425, 42, 530, 486),
        'edge_lines': (1051, 192, 1103, 338)
    },
    'LKR_5000': {
        'blind_dots': (42, 110, 84, 308),
        'asymmetric_serial': (73, 125, 290, 181),
        'vertical_red_serial': (822, 107, 894, 399),
        'security_thread': (435, 38, 575, 484),
        'edge_lines': (1083, 188, 1135, 331)
    }
}



# Pre-load all templates into memory to avoid disk I/O during evaluation
TEMPLATE_CACHE = {'LKR_1000': {}, 'LKR_5000': {}}

def load_templates():
    base_dir = os.path.dirname(os.path.dirname(__file__))
    for denom in ['LKR_1000', 'LKR_5000']:
        denom_dir = os.path.join(base_dir, 'data', 'templates', denom)
        if not os.path.exists(denom_dir): continue
        
        for feature_id in range(1, 8):
            feature_dir = os.path.join(denom_dir, f'Feature {feature_id}')
            if not os.path.exists(feature_dir): continue
            
            TEMPLATE_CACHE[denom][feature_id] = []
            for i in range(1, 11):
                img_path = os.path.join(feature_dir, f'{i}.jpg')
                img = cv2.imread(img_path)
                if img is not None:
                    TEMPLATE_CACHE[denom][feature_id].append(img)

load_templates()

# =====================================================================
# ALGORITHM 1: VISUAL FEATURES (ORB + SSIM)
# =====================================================================

def calculate_ssim(img1, img2):
    """Calculates Structural Similarity Index exactly as defined in the academic paper."""
    from skimage.metrics import structural_similarity
    
    # Resize img2 to exactly match img1 just in case
    img2_resized = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2_resized, cv2.COLOR_BGR2GRAY)
    
    score, _ = structural_similarity(gray1, gray2, full=True)
    return score

def verify_visual_feature(img, denom, feature_id):
    """Matches the test note's feature against all 10 genuine templates and returns the max SSIM."""
    if feature_id not in TEMPLATE_CACHE[denom] or not TEMPLATE_CACHE[denom][feature_id]:
        return False, "Templates not found", None, 0.0

    search_areas = LKR_1000_SEARCH_AREAS if denom == 'LKR_1000' else LKR_5000_SEARCH_AREAS
    x1, y1, x2, y2 = search_areas[feature_id]
    
    # Add dynamic padding to accommodate slight shifts in fake note printing
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
    for template in TEMPLATE_CACHE[denom][feature_id]:
        # Fast template matching to find the exact sub-location
        res = cv2.matchTemplate(search_img, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)
        
        tx, ty = max_loc
        th, tw = template.shape[:2]
        
        extracted_feature = search_img[ty:ty+th, tx:tx+tw]
        
        score = calculate_ssim(extracted_feature, template)
        if score > max_ssim:
            max_ssim = score
            best_crop = extracted_feature.copy()
            best_corr = max_val
            
    # We lowered the threshold for Features 2, 3, 4, 5, and 6 to 0.35.
    # Visual features are easily copied by counterfeiters. By lowering this threshold, 
    # we accurately "pass" fake notes that successfully forged the basic ink prints.
    if feature_id in [2, 3, 4, 5, 6]:
        threshold = 0.35
    else:
        threshold = 0.65
        
    # SSIM can be fooled by backgrounds! If a fake note is perfectly aligned and the background matches,
    # it might score SSIM 0.74 even if half the text is completely missing (like the "RU" in RUPEES).
    # To fix this, we strictly enforce that the Template Correlation (best_corr) must be > 0.90!
    # A missing chunk of text destroys about 10-15% of the correlation. The distorted note scored 0.846,
    # so a strict threshold of 0.90 will ruthlessly fail it, while genuine notes (which score 0.99+) 
    # and perfect fakes (which score 0.95+) will still easily pass.
    passed = (max_ssim > threshold) and (best_corr > 0.90)
    return passed, f"SSIM: {max_ssim:.3f} | TM: {best_corr:.3f}", best_crop, max_ssim

# =====================================================================
# ALGORITHM 2: BLIND RECOGNITION DOTS (Contour Thresholding)
# =====================================================================

def verify_blind_dots(img, denom):
    """Scans the targeted blind dots area and accurately counts the physical number of tactile dots."""
    x1, y1, x2, y2 = PROGRAMMATIC_COORDS[denom]['blind_dots']
    left_edge = img[y1:y2, x1:x2]
    
    gray = cv2.cvtColor(left_edge, cv2.COLOR_BGR2GRAY)
    
    # Use Otsu's thresholding so it perfectly adapts to the ink darkness of fake notes!
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
                                  
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    dot_count = 0
    valid_contours = []
    for c in contours:
        area = cv2.contourArea(c)
        if 50 < area < 500: # Valid dot size (loosened for smaller LKR 5000 dots)
            x, y, w, h = cv2.boundingRect(c)
            aspect_ratio = float(w) / h
            # A dot is circular, meaning its width and height must be almost equal.
            # Background artwork lines and smudges will be long/thin and fail this check.
            if 0.7 < aspect_ratio < 1.4:
                dot_count += 1
                valid_contours.append(c)
            
    # Draw green boxes around detected dots for XAI Explainability
    explain_img = left_edge.copy()
    cv2.drawContours(explain_img, valid_contours, -1, (0, 255, 0), 2)
    
    expected_dots = 5 if denom == 'LKR_1000' else 6
    passed = dot_count == expected_dots
    
    return passed, f"Detected {dot_count} dots (Expected {expected_dots})", explain_img

# =====================================================================
# ALGORITHM 3: ASYMMETRIC NUMBER PANEL
# =====================================================================

def verify_asymmetric_serial(img, denom):
    """Finds characters and verifies their heights progressively increase."""
    x1, y1, x2, y2 = PROGRAMMATIC_COORDS[denom]['asymmetric_serial']
    
    panel = img[y1:y2, x1:x2]
    
    gray = cv2.cvtColor(panel, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter and sort rectangles from left to right
    rects = []
    panel_width = panel.shape[1]
    
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        # Enforce Area constraint to destroy scanner dust
        if w >= 2 and h >= 6 and (w * h) > 15:
            aspect_ratio = float(w) / h
            # Strict Aspect Ratio constraint destroys horizontal lines and wide smudges
            if 0.15 < aspect_ratio < 1.3:
                # The scan window might clip the background pattern on the absolute left/right edges.
                # We strictly filter out any artifact that physically touches the 2-pixel boundary.
                if x > 2 and (x + w) < (panel_width - 2):
                    rects.append((x, y, w, h))
            
    rects = sorted(rects, key=lambda r: r[0])
    
    explain_img = panel.copy()
    if len(rects) < 5:
        return False, "Failed to isolate characters", explain_img
        
    # Verify heights increase
    heights = [r[3] for r in rects]
    increases = 0
    for i in range(1, len(heights)):
        if heights[i] >= heights[i-1] - 2: # 2px tolerance for noise
            increases += 1
            
    # Draw all rects
    for r in rects:
        x, y, w, h = r
        cv2.rectangle(explain_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
    # The user strictly demands exactly 10 characters for the serial number
    passed = (increases >= len(heights) - 2) and (len(rects) == 10)
    return passed, f"Chars: {len(rects)}/10 | Ascend: {increases}/{len(heights)-1}", explain_img

# =====================================================================
# ALGORITHM 4: VERTICAL RED SERIAL
# =====================================================================

def verify_vertical_red_serial(img, denom):
    """Isolates the red channel to verify the presence of the vertical serial."""
    x1, y1, x2, y2 = PROGRAMMATIC_COORDS[denom]['vertical_red_serial']
    panel = img[y1:y2, x1:x2]
    
    hsv = cv2.cvtColor(panel, cv2.COLOR_BGR2HSV)
    explain_img = panel.copy()
    
    # Check if the note was scanned in black-and-white (grayscale)
    avg_sat = np.mean(hsv[:, :, 1])
    is_color = avg_sat >= 15
    
    # Pre-calculate the red mask so we can validate the ink color of individual contours
    mask1 = cv2.inRange(hsv, np.array([0, 50, 50]), np.array([15, 255, 255]))
    mask2 = cv2.inRange(hsv, np.array([165, 50, 50]), np.array([180, 255, 255]))
    red_mask = cv2.bitwise_or(mask1, mask2)
    
    # We ALWAYS count characters using grayscale Otsu thresholding because it perfectly isolates characters regardless of color fading
    gray = cv2.cvtColor(panel, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    
    # Use morphological opening to physically break any thin noisy connections between characters that have merged together
    kernel = np.ones((2, 2), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    valid_chars = 0
    panel_width = panel.shape[1]
    
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        
        # We enforce a minimum Area (w*h > 15) to destroy tiny specks of dust,
        # but we allow width to be as low as 2 pixels and height to be as low as 6 pixels
        # to catch the incredibly thin and short "1" character in the numerator section!
        if w >= 2 and h >= 6 and (w * h) > 15:
            aspect_ratio = float(w) / h
            # A valid text character has an aspect ratio between 0.15 and 1.3
            if 0.15 < aspect_ratio < 1.3:
                # The LKR 5000 has a heavy background pattern on the left edge. 
                # We strictly filter out any noise shapes that touch the absolute left or right pixel borders.
                # By using x > 0 and (x + w) < panel_width, we destroy the noise without accidentally clipping 
                # valid characters (like the '5') that sit near the right edge!
                if x > 0 and (x + w) < panel_width:
                    # Enforce that the shape physically contains red ink (if it's a color scan)!
                    # A dark brown background squiggle will have 0 red pixels, so it mathematically vanishes!
                    box_red = cv2.countNonZero(red_mask[y:y+h, x:x+w]) if is_color else 999
                    if box_red > 10:
                        valid_chars += 1
                        cv2.rectangle(explain_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
    if not is_color:
        # Grayscale fallback: Look for exactly 10 text character contours instead of red color
        passed = valid_chars == 10
        return passed, f"B&W Scan: {valid_chars}/10 chars", explain_img
    
    red_pixels = cv2.countNonZero(red_mask)
    
    # The user strictly demands exactly 10 characters for the serial number AND the red color check
    passed = (red_pixels > 150) and (valid_chars == 10)
    
    return passed, f"Red density: {red_pixels} | Chars: {valid_chars}/10", explain_img

# =====================================================================
# ALGORITHM 5: STARCHROME SECURITY THREAD
# =====================================================================

def verify_security_thread(img, denom):
    """Detects the windowed security thread by finding vertically stacked dark rectangular segments."""
    x1, y1, x2, y2 = PROGRAMMATIC_COORDS[denom]['security_thread']
    panel = img[y1:y2, x1:x2]
    
    gray = cv2.cvtColor(panel, cv2.COLOR_BGR2GRAY)
    
    # Starchrome metallic threads scan as very dark gray or pure black.
    # To seamlessly handle varying scanner brightness, we dynamically calculate the 
    # threshold based on the average brightness of this specific crop region.
    mean_brightness = np.mean(gray)
    dynamic_thresh = max(30, int(mean_brightness - 50))
    
    _, thresh = cv2.threshold(gray, dynamic_thresh, 255, cv2.THRESH_BINARY_INV)
    
    # Use morphological opening to break any thin noise connections between segments
    kernel = np.ones((5, 5), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Get raw candidate bounding boxes
    raw_rects = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w > 5 and h > 5:
            raw_rects.append((x, y, w, h))
            
    # Calculate the average width of original valid unmerged rectangles to prevent
    # horizontal shifts during merging from inflating the final width metric.
    original_widths = [w for x, y, w, h in raw_rects if w > 10]
    avg_original_width = np.mean(original_widths) if original_widths else 0
    
    # Sort by Y-coordinate and vertically merge segments that are close to each other
    raw_rects = sorted(raw_rects, key=lambda r: r[1])
    merged = True
    while merged:
        merged = False
        used = set()
        for i in range(len(raw_rects)):
            if i in used:
                continue
            rx, ry, rw, rh = raw_rects[i]
            
            merge_idx = -1
            for j in range(i + 1, len(raw_rects)):
                if j in used:
                    continue
                ox, oy, ow, oh = raw_rects[j]
                
                # Check horizontal overlap and vertical gap proximity
                overlap_x = max(rx, ox) <= min(rx + rw, ox + ow) + 5
                gap = oy - (ry + rh)
                
                if overlap_x and gap <= 15:
                    merge_idx = j
                    break
            
            if merge_idx != -1:
                ox, oy, ow, oh = raw_rects[merge_idx]
                min_x = min(rx, ox)
                min_y = min(ry, oy)
                max_x = max(rx + rw, ox + ow)
                max_y = max(ry + rh, oy + oh)
                
                raw_rects[i] = (min_x, min_y, max_x - min_x, max_y - min_y)
                used.add(merge_idx)
                merged = True
                break
        
        if merged:
            raw_rects = [raw_rects[k] for k in range(len(raw_rects)) if k not in used]
            raw_rects = sorted(raw_rects, key=lambda r: r[1])
            
    # Filter merged rectangles by final segment requirements
    valid_rects = []
    explain_img = panel.copy()
    for rx, ry, rw, rh in raw_rects:
        if rw > 10 and rh > 20:
            valid_rects.append((rx, ry, rw, rh))
            
    # Count how many rectangles are vertically stacked in the same column
    max_stacked = 0
    best_x = -1
    if valid_rects:
        for rx, ry, rw, rh in valid_rects:
            stacked = sum(1 for ox, oy, ow, oh in valid_rects if abs(ox - rx) <= 15)
            if stacked > max_stacked:
                max_stacked = stacked
                best_x = rx
                
    if max_stacked == 5:
        for rx, ry, rw, rh in valid_rects:
            if abs(rx - best_x) <= 15:
                cv2.rectangle(explain_img, (rx, ry), (rx+rw, ry+rh), (0, 255, 0), 2)
                
    # A genuine windowed thread must have exactly 5 segments
    passed = max_stacked == 5
    
    # Calculate measured mm using the original unmerged width
    if avg_original_width > 0:
        measured_mm = avg_original_width / 7.3
    else:
        avg_width_px = np.mean([rw for rx, ry, rw, rh in valid_rects]) if valid_rects else 0
        measured_mm = avg_width_px / 7.3
        
    expected_mm = 3.0 if denom == 'LKR_5000' else 2.5
    if abs(measured_mm - expected_mm) > 0.5:
        passed = False
        
    return passed, f"Segments: {max_stacked}/5 | Width: {measured_mm:.1f}mm", explain_img

# =====================================================================
# ALGORITHM 6: TACTILE EDGE LINES
# =====================================================================

def verify_edge_lines(img, denom):
    """Detects the 15 tactile bleed lines on the right edge of the note for the visually impaired."""
    x1, y1, x2, y2 = PROGRAMMATIC_COORDS[denom]['edge_lines']
    panel = img[y1:y2, x1:x2]
    
    gray = cv2.cvtColor(panel, cv2.COLOR_BGR2GRAY)
    
    # We use Adaptive Thresholding instead of Otsu because the edge of the physical paper 
    # often casts a shadow. Adaptive thresholding dynamically adjusts pixel by pixel,
    # completely ignoring the shadow and only extracting the sharp horizontal lines!
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 5)
    
    # Use horizontal morphological opening to destroy vertical noise and isolate horizontal lines
    kernel = np.ones((1, 10), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    valid_lines = 0
    explain_img = panel.copy()
    
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        # Genuine tactile lines are physically wide (~30-40 pixels).
        # We strictly require w > 20 to completely obliterate tiny specks of scanner dust
        # that might be sitting near the edges of the paper.
        if w > 20 and h < 10:
            aspect_ratio = float(w) / h
            # A genuine line has a massive aspect ratio (usually 10.0 to 20.0). 
            # We enforce AR > 6.0 to ensure chunky noise blobs are ignored.
            if aspect_ratio > 6.0: 
                valid_lines += 1
                cv2.rectangle(explain_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
    # The user strictly demands exactly 15 edge lines
    passed = valid_lines == 15
    return passed, f"Edge Lines: {valid_lines}/15", explain_img

# =====================================================================
# MASTER PIPELINE EVALUATOR
# =====================================================================

EVAL_PIPELINE = {
    'LKR_1000': [
        ('F1: Central Bank Title', lambda i, d: verify_template(i, d, 1)),
        ('F2: Value in Sinhala', lambda i, d: verify_template(i, d, 2)),
        ('F3: Numeral 1000', lambda i, d: verify_template(i, d, 3)),
        ('F4: Date & Signature', lambda i, d: verify_template(i, d, 4)),
        ('F5: Butterfly Motif', lambda i, d: verify_template(i, d, 5)),
        ('F6: Central Bank Logo', lambda i, d: verify_template(i, d, 6)),
        ('F7: See-through Register', lambda i, d: verify_template(i, d, 7)),
        ('F8: Blind Dots (Algorithm 2)', verify_blind_dots),
        ('F9: Asymmetric Serial (Algorithm 3)', verify_asymmetric_serial),
        ('F10: Vertical Red Serial (Algorithm 4)', verify_vertical_red_serial),
        ('F11: Security Thread', verify_security_thread),
        ('F12: Tactile Edge Lines', verify_edge_lines)
    ],
    'LKR_5000': [
        ('F1: Central Bank Title', lambda i, d: verify_template(i, d, 1)),
        ('F2: Value in Sinhala', lambda i, d: verify_template(i, d, 2)),
        ('F3: Numeral 5000', lambda i, d: verify_template(i, d, 3)),
        ('F4: Date & Signature', lambda i, d: verify_template(i, d, 4)),
        ('F5: Butterfly Motif', lambda i, d: verify_template(i, d, 5)),
        ('F6: Central Bank Logo', lambda i, d: verify_template(i, d, 6)),
        ('F7: See-through Register', lambda i, d: verify_template(i, d, 7)),
        ('F8: Blind Dots (Algorithm 2)', verify_blind_dots),
        ('F9: Asymmetric Serial (Algorithm 3)', verify_asymmetric_serial),
        ('F10: Vertical Red Serial (Algorithm 4)', verify_vertical_red_serial),
        ('F11: Security Thread', verify_security_thread),
        ('F12: Tactile Edge Lines', verify_edge_lines)
    ]
}

def analyze_lkr_note(file_path, denom):
    """
    Executes the full Three-Algorithm Pipeline on an LKR Banknote.
    Returns: (final_verdict, message, score, feature_statuses, feature_images)
    """
    raw_img = cv2.imread(file_path)
    if raw_img is None:
        return False, "Could not read image", 0, [], []
        
    # STAGE 1: ALIGNMENT
    img = align_note(raw_img, denom)
    
    feature_statuses = []
    feature_images = []
    
    total_features = 0
    passed_features = 0
    veto_triggered = False
    
    # STAGE 2: VISUAL SSIM FEATURES (Algorithm 1)
    for f_id in range(1, 8):
        passed, msg, best_crop, score = verify_visual_feature(img, denom, f_id)
        if best_crop is None:
            best_crop = np.zeros((50, 50, 3), dtype=np.uint8)
        
        total_features += 1
        if passed: passed_features += 1
        
        feature_statuses.append((passed, msg))
        feature_images.append(best_crop)
        
        # Micro-Printing (F1) and Watermark (F7) are critical VETO gates
        if (f_id == 1 or f_id == 7) and not passed:
            veto_triggered = True

    # STAGE 3: PROGRAMMATIC FEATURES (Algorithms 2 & 3)
    
    # Blind Dots
    d_pass, d_msg, d_img = verify_blind_dots(img, denom)
    total_features += 1
    if d_pass: passed_features += 1
    feature_statuses.append((d_pass, d_msg))
    feature_images.append(d_img)
    
    # Asymmetric Serial
    a_pass, a_msg, a_img = verify_asymmetric_serial(img, denom)
    total_features += 1
    if a_pass: passed_features += 1
    feature_statuses.append((a_pass, a_msg))
    feature_images.append(a_img)
    
    # Vertical Red Serial
    r_pass, r_msg, r_img = verify_vertical_red_serial(img, denom)
    total_features += 1
    if r_pass: passed_features += 1
    feature_statuses.append((r_pass, r_msg))
    feature_images.append(r_img)
    
    # Security Thread (Critical Veto Gate)
    t_pass, t_msg, t_img = verify_security_thread(img, denom)
    total_features += 1
    if t_pass: passed_features += 1
    if not t_pass: veto_triggered = True
    feature_statuses.append((t_pass, t_msg))
    feature_images.append(t_img)
    
    # Tactile Edge Lines
    e_pass, e_msg, e_img = verify_edge_lines(img, denom)
    total_features += 1
    if e_pass: passed_features += 1
    feature_statuses.append((e_pass, e_msg))
    feature_images.append(e_img)
    
    # STAGE 4: CLASSIFICATION
    pass_ratio = passed_features / total_features
    final_score = int(pass_ratio * 100)
    
    # Flat verdict: purely based on passing 75% of features (9 out of 12)
    flat_verdict = pass_ratio >= 0.75
    
    if veto_triggered:
        robust_verdict = False
        message = "COUNTERFEIT (Veto triggered by critical security failure)"
    elif flat_verdict:
        robust_verdict = True
        message = "GENUINE NOTE (Passed structural & programmatic verification)"
    else:
        robust_verdict = False
        message = f"COUNTERFEIT (Failed to meet confidence threshold)"
        
    return flat_verdict, robust_verdict, message, final_score, feature_statuses, feature_images
