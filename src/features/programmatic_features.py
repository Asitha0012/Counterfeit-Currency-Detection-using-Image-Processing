import cv2
import numpy as np

# =====================================================================
# CONFIGURATION & COORDINATES
# =====================================================================

PROGRAMMATIC_COORDS = {
    'LKR_1000': {
        'blind_dots': (40, 110, 82, 310),
        'asymmetric_serial': (65, 120, 290, 200),
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

# =====================================================================
# ALGORITHM 2: BLIND RECOGNITION DOTS (Contour Thresholding)
# =====================================================================

def verify_blind_dots(img, denom):
    """Scans the targeted blind dots area and accurately counts the physical number of tactile dots."""
    x1, y1, x2, y2 = PROGRAMMATIC_COORDS[denom]['blind_dots']
    left_edge = img[y1:y2, x1:x2]
    
    if left_edge.size == 0:
        return False, "Invalid image bounds", np.zeros((50, 50, 3), dtype=np.uint8)
        
    gray = cv2.cvtColor(left_edge, cv2.COLOR_BGR2GRAY)
    
    blurred = cv2.medianBlur(gray, 5)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 10)
                                  
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    dot_count = 0
    valid_contours = []
    for c in contours:
        area = cv2.contourArea(c)
        if 50 < area < 500: # Valid dot size (loosened for smaller LKR 5000 dots)
            x, y, w, h = cv2.boundingRect(c)
            aspect_ratio = float(w) / h
            if 0.5 < aspect_ratio < 1.6:
                valid_contours.append(c)
                
    # Filter out contours that are not vertically aligned with the majority
    if valid_contours:
        centers_x = [cv2.boundingRect(c)[0] + cv2.boundingRect(c)[2]/2 for c in valid_contours]
        median_x = np.median(centers_x)
        aligned_contours = [c for i, c in enumerate(valid_contours) if abs(centers_x[i] - median_x) < 15]
        valid_contours = aligned_contours
        dot_count = len(valid_contours)
            
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
    
    if panel.size == 0:
        return False, "Invalid image bounds", np.zeros((50, 50, 3), dtype=np.uint8)
        
    gray = cv2.cvtColor(panel, cv2.COLOR_BGR2GRAY)
    blurred = cv2.medianBlur(gray, 5)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 10)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    rects = []
    panel_width = panel.shape[1]
    
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w >= 5 and h >= 10 and (w * h) > 30:
            aspect_ratio = float(w) / h
            if 0.15 < aspect_ratio < 2.5:
                if x > 2 and (x + w) < (panel_width - 2):
                    rects.append((x, y, w, h))
                
    split_rects = []
    for r in rects:
        x, y, w, h = r
        if w / h > 1.3:
            # Split merged characters
            half_w = w // 2
            split_rects.append((x, y, half_w, h))
            split_rects.append((x + half_w, y, w - half_w, h))
        else:
            if w / h > 0.15:
                split_rects.append(r)
                
    explain_img = panel.copy()
    if not split_rects:
        return False, "No characters found", explain_img
        
    largest_temp = sorted(split_rects, key=lambda r: r[2] * r[3], reverse=True)[:5]
    med_y = np.median([r[1] + r[3]/2 for r in largest_temp])
    
    aligned_rects = [r for r in split_rects if abs((r[1] + r[3]/2) - med_y) < 20]
    
    expected_chars = 10
    
    final_rects = sorted(aligned_rects, key=lambda r: r[2] * r[3], reverse=True)[:expected_chars]
    final_rects = sorted(final_rects, key=lambda r: r[0])
    
    if len(final_rects) < 8:
        return False, f"Failed to isolate 8-10 characters (found {len(final_rects)})", explain_img
        
    heights = [r[3] for r in final_rects]
    increases = 0
    for i in range(1, len(heights)):
        if heights[i] >= heights[i-1] - 3:
            increases += 1
            
    for r in final_rects:
        x, y, w, h = r
        cv2.rectangle(explain_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
    passed = (increases >= len(heights) - 2) and (len(final_rects) in [8, 9, 10])
    return passed, f"Chars: {len(final_rects)}/{expected_chars} | Ascend: {increases}/{len(heights)-1}", explain_img

# =====================================================================
# ALGORITHM 4: VERTICAL RED SERIAL
# =====================================================================

def verify_vertical_red_serial(img, denom):
    """Isolates the red channel to verify the presence of the vertical serial."""
    x1, y1, x2, y2 = PROGRAMMATIC_COORDS[denom]['vertical_red_serial']
    panel = img[y1:y2, x1:x2]
    
    if panel.size == 0:
        return False, "Invalid image bounds", np.zeros((50, 50, 3), dtype=np.uint8)
        
    hsv = cv2.cvtColor(panel, cv2.COLOR_BGR2HSV)
    explain_img = panel.copy()
    
    avg_sat = np.mean(hsv[:, :, 1])
    is_color = avg_sat >= 15
    
    mask1 = cv2.inRange(hsv, np.array([0, 50, 50]), np.array([15, 255, 255]))
    mask2 = cv2.inRange(hsv, np.array([165, 50, 50]), np.array([180, 255, 255]))
    red_mask = cv2.bitwise_or(mask1, mask2)
    
    gray = cv2.cvtColor(panel, cv2.COLOR_BGR2GRAY)
    blurred = cv2.medianBlur(gray, 5)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 10)
    
    kernel = np.ones((2, 2), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    valid_chars = 0
    panel_width = panel.shape[1]
    
    rects = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w >= 3 and h >= 6 and (w * h) > 20: # Slightly stricter area filter
            aspect_ratio = float(w) / h
            if 0.15 < aspect_ratio < 1.5:
                if x > 0 and (x + w) < panel_width:
                    box_red = cv2.countNonZero(red_mask[y:y+h, x:x+w]) if is_color else 999
                    if box_red > 10:
                        rects.append((x, y, w, h))
                        
    expected_chars = 10
    
    # Top expected_chars largest by area to exclude background noise/speckles
    rects = sorted(rects, key=lambda r: r[2] * r[3], reverse=True)[:expected_chars]
    valid_chars = len(rects)
    
    for r in rects:
        x, y, w, h = r
        cv2.rectangle(explain_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
    if not is_color:
        passed = valid_chars in [8, 9, 10]
        return passed, f"B&W Scan: {valid_chars}/{expected_chars} chars", explain_img
    
    red_pixels = cv2.countNonZero(red_mask)
    passed = (red_pixels > 150) and (valid_chars in [8, 9, 10])
    
    return passed, f"Red density: {red_pixels} | Chars: {valid_chars}/{expected_chars}", explain_img

# =====================================================================
# ALGORITHM 5: STARCHROME SECURITY THREAD
# =====================================================================

def verify_security_thread(img, denom):
    """Detects the windowed security thread by finding vertically stacked dark rectangular segments."""
    x1, y1, x2, y2 = PROGRAMMATIC_COORDS[denom]['security_thread']
    panel = img[y1:y2, x1:x2]
    
    if panel.size == 0:
        return False, "Invalid image bounds", np.zeros((50, 50, 3), dtype=np.uint8)
        
    gray = cv2.cvtColor(panel, cv2.COLOR_BGR2GRAY)
    
    mean_brightness = np.mean(gray)
    dynamic_thresh = max(30, int(mean_brightness - 50))
    
    _, thresh = cv2.threshold(gray, dynamic_thresh, 255, cv2.THRESH_BINARY_INV)
    
    kernel = np.ones((5, 5), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    raw_rects = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w > 5 and h > 5:
            raw_rects.append((x, y, w, h))
            
    original_widths = [w for x, y, w, h in raw_rects if w > 10]
    avg_original_width = np.mean(original_widths) if original_widths else 0
    
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
            
    valid_rects = []
    explain_img = panel.copy()
    for rx, ry, rw, rh in raw_rects:
        if rw > 10 and rh >= 10:
            valid_rects.append((rx, ry, rw, rh))
            
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
                
    passed = max_stacked == 5
    
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
    
    if panel.size == 0:
        return False, "Invalid image bounds", np.zeros((50, 50, 3), dtype=np.uint8)
        
    gray = cv2.cvtColor(panel, cv2.COLOR_BGR2GRAY)
    
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 5)
    
    kernel = np.ones((1, 10), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    valid_lines = 0
    explain_img = panel.copy()
    
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w > 20 and h < 10:
            aspect_ratio = float(w) / h
            if aspect_ratio > 6.0: 
                valid_lines += 1
                cv2.rectangle(explain_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
    passed = valid_lines == 15
    return passed, f"Edge Lines: {valid_lines}/15", explain_img
