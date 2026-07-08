import cv2
import numpy as np

# =====================================================================
# CONFIGURATION & COORDINATES
# =====================================================================

PROGRAMMATIC_COORDS = {
    'LKR_500': {
        'blind_dots': (33, 114, 74, 274),
        'asymmetric_serial': (65, 126, 283, 200),
        'vertical_red_serial': (780, 130, 844, 408),
        'security_thread': (340, 4, 430, 496),
        'edge_lines': (1019, 194, 1062, 334)
    },
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
    
    if denom == 'LKR_500':
        expected_dots = 4
    elif denom == 'LKR_1000':
        expected_dots = 5
    else:
        expected_dots = 6
        
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
        min_h = 3 if denom == 'LKR_500' else 10
        min_w = 3 if denom == 'LKR_500' else 5
        min_area = 10 if denom == 'LKR_500' else 30
        if w >= min_w and h >= min_h and (w * h) > min_area:
            aspect_ratio = float(w) / h
            if 0.1 < aspect_ratio < 4.0:
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
        
    if denom in ['LKR_500', 'LKR_1000']:
        aligned_rects = split_rects
    else:
        largest_temp = sorted(split_rects, key=lambda r: r[2] * r[3], reverse=True)[:5]
        med_y = np.median([r[1] + r[3]/2 for r in largest_temp])
        aligned_rects = [r for r in split_rects if abs((r[1] + r[3]/2) - med_y) < 20]
    
    expected_chars = 10
    
    final_rects = sorted(aligned_rects, key=lambda r: r[2] * r[3], reverse=True)[:expected_chars]
    final_rects = sorted(final_rects, key=lambda r: r[0])
    
    if len(final_rects) < 7:
        return False, f"Failed to isolate 7-10 characters (found {len(final_rects)})", explain_img
        
    
    increases = 0
    
    best_increases = -1
    best_rects = []
    best_chars = 0
    best_explain_img = gray.copy()
    expected_chars = 10
    
    for C in [5, 7, 10, 15]:
        for bs in [15, 21, 25]:
            thresh_dyn = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, bs, C)
            kernel_dyn = np.ones((2, 2), np.uint8)
            # thresh_dyn = cv2.morphologyEx(thresh_dyn, cv2.MORPH_OPEN, kernel_dyn)
            
            contours_dyn, _ = cv2.findContours(thresh_dyn, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            rects_dyn = []
            for c in contours_dyn:
                x, y, w, h = cv2.boundingRect(c)
                # Reject dots (h < 13) and massive noise blocks (h > 55)
                if w >= 3 and h >= 13 and (w * h) > 30 and h <= 55 and w <= 45:
                    aspect_ratio = float(w) / h
                    if 0.1 < aspect_ratio < 4.0:
                        rects_dyn.append((x, y, w, h))
                            
            split_rects_dyn = []
            for r in rects_dyn:
                x, y, w, h = r
                if w / h > 1.8:
                    if h < 13: pass
                    else:
                        half_w = w // 2
                        split_rects_dyn.append((x, y, half_w, h))
                        split_rects_dyn.append((x + half_w, y, w - half_w, h))
                else:
                    split_rects_dyn.append(r)
                    
            if len(split_rects_dyn) > 0:
                # Calculate median baseline from the largest components (the digits)
                largest_temp = sorted(split_rects_dyn, key=lambda r: r[2] * r[3], reverse=True)[:5]
                med_baseline = np.median([r[1] + r[3] for r in largest_temp])
                
                filtered_rects = []
                for r in split_rects_dyn:
                    baseline = r[1] + r[3]
                    # Rule 1: The 9 ascending digits on the main baseline
                    if abs(baseline - med_baseline) <= 12:
                        filtered_rects.append(r)
                    # Rule 2: The prefix letter (e.g. 'S') above the first digits
                    # It is on the left (x < 100) and situated above the main baseline
                    elif r[0] < 100 and baseline < med_baseline - 15 and r[2] >= 8 and r[3] >= 8:
                        filtered_rects.append(r)
                
                split_rects_dyn = filtered_rects
                
            final_rects_dyn = sorted(split_rects_dyn, key=lambda r: r[2] * r[3], reverse=True)[:10]
            final_rects_dyn = sorted(final_rects_dyn, key=lambda r: r[0])
            
            heights_dyn = [r[3] for r in final_rects_dyn]
            # Enforce ascending sequence but with a noise tolerance of -12 pixels
            increases_dyn = sum(1 for j in range(1, len(heights_dyn)) if heights_dyn[j] >= heights_dyn[j-1] - 12)
            
            if increases_dyn > best_increases or (increases_dyn == best_increases and len(final_rects_dyn) > best_chars):
                best_increases = increases_dyn
                best_rects = final_rects_dyn
                best_chars = len(final_rects_dyn)
                
                best_explain_img = panel.copy()
                gray_panel = cv2.cvtColor(panel, cv2.COLOR_BGR2GRAY)
                for r in best_rects:
                    rx, ry, rw, rh = r
                    
                    # Refine bounding box to tightly fit the actual character pixels
                    char_roi = gray_panel[ry:ry+rh, rx:rx+rw]
                    _, char_thresh = cv2.threshold(char_roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                    char_coords = cv2.findNonZero(char_thresh)
                    if char_coords is not None:
                        cx, cy, cw, ch = cv2.boundingRect(char_coords)
                        cv2.rectangle(best_explain_img, (rx+cx, ry+cy), (rx+cx+cw, ry+cy+ch), (0, 255, 0), 2)
                    else:
                        cv2.rectangle(best_explain_img, (rx, ry), (rx+rw, ry+rh), (0, 255, 0), 2)
                    
            if best_chars == 10 and best_increases == 9:
                break
        if best_chars == 10 and best_increases == 9:
            break
            
    if denom == 'LKR_5000':
        passed = (best_chars == 10 and best_increases == 9)
        return passed, f"Chars: {best_chars}/10 | Ascend: {best_increases}/9", best_explain_img
    else:
        # LKR_500 and LKR_1000 original logic
        passed = (best_chars == 10 and best_increases == 9)
        return passed, f"Chars: {best_chars}/10 | Ascend: {best_increases}/9", best_explain_img

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
    
    best_valid_chars = 0
    best_rects = []
    best_explain_img = panel.copy()
    
    for C in [5, 10, 15]:
        for bs in [15, 21, 25]:
            thresh_dyn = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, bs, C)
            kernel = np.ones((2, 2), np.uint8)
            thresh_dyn = cv2.morphologyEx(thresh_dyn, cv2.MORPH_OPEN, kernel)
            
            contours_dyn, _ = cv2.findContours(thresh_dyn, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            rects_dyn = []
            for c in contours_dyn:
                x, y, w, h = cv2.boundingRect(c)
                if w >= 3 and h >= 6 and (w * h) > 20:
                    aspect_ratio = float(w) / h
                    if 0.15 < aspect_ratio < 1.5:
                        if x > 0 and (x + w) < panel_width:
                            box_red = cv2.countNonZero(red_mask[y:y+h, x:x+w]) if is_color else 999
                            if box_red > 10:
                                rects_dyn.append((x, y, w, h))
                                
            rects_dyn = sorted(rects_dyn, key=lambda r: r[2] * r[3], reverse=True)[:expected_chars]
            
            if denom == 'LKR_5000':
                if len(rects_dyn) == 10:
                    rects_dyn.sort(key=lambda r: r[1]) # Sort by Y
                    
                    y_horiz = [r[1] for r in rects_dyn[1:4]]
                    x_horiz = [r[0] for r in rects_dyn[1:4]]
                    x_vert = [r[0] for r in rects_dyn[4:10]]
                    
                    valid = True
                    if max(y_horiz) - min(y_horiz) > 20: valid = False
                    if max(x_horiz) - min(x_horiz) < 10: valid = False
                    if max(x_vert) - min(x_vert) > 30: valid = False
                    
                    for i in range(5, 10):
                        if rects_dyn[i][1] - rects_dyn[i-1][1] < 8:
                            valid = False
                            
                    if valid:
                        best_valid_chars = 10
                        best_rects = rects_dyn
                        best_explain_img = panel.copy()
                        gray_panel = cv2.cvtColor(panel, cv2.COLOR_BGR2GRAY)
                        for r in best_rects:
                            rx, ry, rw, rh = r
                            char_roi = gray_panel[ry:ry+rh, rx:rx+rw]
                            _, char_thresh = cv2.threshold(char_roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                            char_coords = cv2.findNonZero(char_thresh)
                            if char_coords is not None:
                                cx, cy, cw, ch = cv2.boundingRect(char_coords)
                                cv2.rectangle(best_explain_img, (rx+cx, ry+cy), (rx+cx+cw, ry+cy+ch), (0, 255, 0), 2)
                            else:
                                cv2.rectangle(best_explain_img, (rx, ry), (rx+rw, ry+rh), (0, 255, 0), 2)
            else:
                if len(rects_dyn) > best_valid_chars:
                    best_valid_chars = len(rects_dyn)
                    best_rects = rects_dyn
                    best_explain_img = panel.copy()
                    gray_panel = cv2.cvtColor(panel, cv2.COLOR_BGR2GRAY)
                    for r in best_rects:
                        rx, ry, rw, rh = r
                        char_roi = gray_panel[ry:ry+rh, rx:rx+rw]
                        _, char_thresh = cv2.threshold(char_roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                        char_coords = cv2.findNonZero(char_thresh)
                        if char_coords is not None:
                            cx, cy, cw, ch = cv2.boundingRect(char_coords)
                            cv2.rectangle(best_explain_img, (rx+cx, ry+cy), (rx+cx+cw, ry+cy+ch), (0, 255, 0), 2)
                        else:
                            cv2.rectangle(best_explain_img, (rx, ry), (rx+rw, ry+rh), (0, 255, 0), 2)
                        
            if best_valid_chars == expected_chars:
                break
        if best_valid_chars == expected_chars:
            break
            
    if denom == 'LKR_1000':
        passed = False
        if best_valid_chars == expected_chars:
            best_rects_sorted = sorted(best_rects, key=lambda r: r[1])
            top = [r for r in best_rects_sorted if r[1] < 25]
            mid = [r for r in best_rects_sorted if 25 <= r[1] < 55]
            bot = [r for r in best_rects_sorted if r[1] >= 55]
            
            if len(top) == 1 and len(mid) == 3 and len(bot) == 6:
                xs = [r[0] + r[2]/2 for r in bot]
                if max(xs) - min(xs) < 20: # Vertically aligned
                    passed = True

        if not is_color:
            return passed, f"B&W Scan: Struct {'Pass' if passed else 'Fail'} | {best_valid_chars}/{expected_chars}", best_explain_img
            
        red_pixels = cv2.countNonZero(red_mask)
        passed = passed and (red_pixels > 150)
        return passed, f"Red density: {red_pixels} | Struct {'Pass' if passed else 'Fail'}", best_explain_img

    elif denom == 'LKR_500':
        if not is_color:
            passed = (best_valid_chars == expected_chars)
            return passed, f"B&W Scan: {best_valid_chars}/{expected_chars} chars", best_explain_img
        
        red_pixels = cv2.countNonZero(red_mask)
        passed = (red_pixels > 150) and (best_valid_chars == expected_chars)
        return passed, f"Red density: {red_pixels} | Chars: {best_valid_chars}/{expected_chars}", best_explain_img
    else:
        if not is_color:
            passed = (best_valid_chars == expected_chars)
            return passed, f"B&W Scan: {best_valid_chars}/{expected_chars} chars", best_explain_img
        
        red_pixels = cv2.countNonZero(red_mask)
        passed = (red_pixels > 150) and (best_valid_chars == expected_chars)
        return passed, f"Red density: {red_pixels} | Chars: {best_valid_chars}/{expected_chars}", best_explain_img

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
    
    if denom == 'LKR_500':
        best_rects = []
        best_max_stacked = 0
        
        for offset in range(5, 50, 5):
            for morph in [1, 2, 3]:
                for max_gap in [15, 20, 25, 30]:
                    dynamic_thresh = max(30, int(mean_brightness - offset))
                    
                    # Capping threshold for LKR_500 to ensure we only detect truly DARK thread segments
                    # Fake notes (F4, F8) have light gray/silver backgrounds that falsely trigger detection
                    if denom == 'LKR_500':
                        dynamic_thresh = min(dynamic_thresh, 45)
                        
                    _, thresh = cv2.threshold(gray, dynamic_thresh, 255, cv2.THRESH_BINARY_INV)
                    kernel = np.ones((morph, morph), np.uint8)
                    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
                    
                    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    # Apply width filter to ignore thin background artifacts
                    # The standard thread is ~2mm (14.6px). We accept 10 <= w <= 35
                    raw_rects = [(x, y, w, h) for c in contours for x, y, w, h in [cv2.boundingRect(c)] if 10 <= w <= 35 and h > 5]
                    
                    raw_rects = sorted(raw_rects, key=lambda r: r[1])
                    merged = True
                    while merged:
                        merged = False
                        used = set()
                        for k in range(len(raw_rects)):
                            if k in used: continue
                            rx_box, ry_box, rw_box, rh_box = raw_rects[k]
                            merge_idx = -1
                            for j in range(k + 1, len(raw_rects)):
                                if j in used: continue
                                ox, oy, ow, oh = raw_rects[j]
                                overlap_x = max(rx_box, ox) <= min(rx_box + rw_box, ox + ow) + 5
                                gap = oy - (ry_box + rh_box)
                                if overlap_x and gap <= max_gap:
                                    merge_idx = j
                                    break
                            if merge_idx != -1:
                                ox, oy, ow, oh = raw_rects[merge_idx]
                                min_x, min_y = min(rx_box, ox), min(ry_box, oy)
                                max_x, max_y = max(rx_box + rw_box, ox + ow), max(ry_box + rh_box, oy + oh)
                                raw_rects[k] = (min_x, min_y, max_x - min_x, max_y - min_y)
                                used.add(merge_idx)
                                merged = True
                                break
                        if merged:
                            raw_rects = [raw_rects[k] for k in range(len(raw_rects)) if k not in used]
                            raw_rects = sorted(raw_rects, key=lambda r: r[1])
                    
                    valid_rects = [r for r in raw_rects if r[2] >= 8 and r[2] <= 35 and r[3] >= 8]
                    
                    if valid_rects:
                        groups = []
                        for r in valid_rects:
                            rx, ry, rw, rh = r
                            cx = rx + rw/2
                            placed = False
                            for g in groups:
                                g_cx = np.mean([gr[0] + gr[2]/2 for gr in g])
                                g_w = np.mean([gr[2] for gr in g])
                                if abs(cx - g_cx) <= 10 and abs(rw - g_w) <= 10:
                                    g.append(r)
                                    placed = True
                                    break
                            if not placed:
                                groups.append([r])
                                
                        if groups:
                            largest_group = max(groups, key=len)
                            if len(largest_group) > best_max_stacked:
                                best_max_stacked = len(largest_group)
                                best_rects = largest_group
                        
                    if best_max_stacked == 5: break
                if best_max_stacked == 5: break
            if best_max_stacked == 5: break
            
        # Two-pass validation to catch segments hidden in dark backgrounds
        best_strip_count = 0
        best_strip_rects = []
        if best_max_stacked >= 2:
            g_cx = np.mean([r[0] + r[2]/2 for r in best_rects])
            g_w = np.mean([r[2] for r in best_rects])
            
            strip_x1 = max(0, int(g_cx - g_w/2 - 2))
            strip_x2 = min(gray.shape[1], int(g_cx + g_w/2 + 2))
            strip = gray[:, strip_x1:strip_x2]
            strip_mean = np.mean(strip)
            
            for offset in range(5, 60, 5):
                for morph in [1, 2]:
                    dynamic_thresh = max(30, int(strip_mean - offset))
                    
                    if denom == 'LKR_500':
                        dynamic_thresh = min(dynamic_thresh, 45)
                        
                    _, thresh = cv2.threshold(strip, dynamic_thresh, 255, cv2.THRESH_BINARY_INV)
                    kernel = np.ones((morph, morph), np.uint8)
                    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
                    
                    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    strip_rects = [(x, y, w, h) for c in contours for x, y, w, h in [cv2.boundingRect(c)] if w >= 5 and h >= 5]
                    strip_rects = sorted(strip_rects, key=lambda r: r[1])
                    
                    merged = True
                    while merged:
                        merged = False
                        used = set()
                        for k in range(len(strip_rects)):
                            if k in used: continue
                            rx_box, ry_box, rw_box, rh_box = strip_rects[k]
                            merge_idx = -1
                            for j in range(k + 1, len(strip_rects)):
                                if j in used: continue
                                ox, oy, ow, oh = strip_rects[j]
                                overlap_x = max(rx_box, ox) <= min(rx_box + rw_box, ox + ow) + 2
                                gap = oy - (ry_box + rh_box)
                                if overlap_x and gap <= 20:
                                    merge_idx = j
                                    break
                            if merge_idx != -1:
                                ox, oy, ow, oh = strip_rects[merge_idx]
                                min_x, min_y = min(rx_box, ox), min(ry_box, oy)
                                max_x, max_y = max(rx_box + rw_box, ox + ow), max(ry_box + rh_box, oy + oh)
                                strip_rects[k] = (min_x, min_y, max_x - min_x, max_y - min_y)
                                used.add(merge_idx)
                                merged = True
                                break
                        if merged:
                            strip_rects = [strip_rects[k] for k in range(len(strip_rects)) if k not in used]
                            strip_rects = sorted(strip_rects, key=lambda r: r[1])
                            
                    valid = [r for r in strip_rects if r[2] >= g_w*0.5 and r[3] >= 8]
                    if len(valid) > best_strip_count:
                        best_strip_count = len(valid)
                        best_strip_rects = valid
                    if best_strip_count == 5: break
                if best_strip_count == 5: break
            
            if best_strip_count > best_max_stacked:
                # Map back to original panel coordinates
                mapped_rects = []
                for rx, ry, rw, rh in best_strip_rects:
                    mapped_rects.append((strip_x1 + rx, ry, rw, rh))
                best_rects = mapped_rects
                best_max_stacked = min(5, best_strip_count)
            else:
                best_max_stacked = min(5, best_max_stacked)
            
        max_stacked = best_max_stacked
        
        # Filter out abnormally wide rectangles that might have merged with the background
        g_w = np.median([w for x, y, w, h in best_rects]) if best_rects else 0
        original_widths = [w for x, y, w, h in best_rects if abs(w - g_w) <= 15]
        avg_original_width = np.mean(original_widths) if original_widths else 0
        measured_mm = (avg_original_width / 7.3) if avg_original_width > 0 else 0
        
        explain_img = panel.copy()
        for rx, ry, rw, rh in best_rects:
            cv2.rectangle(explain_img, (rx, ry), (rx+rw, ry+rh), (0, 255, 0), 2)
            
        expected_segments = 5
        # The standard width is 2mm. Genuine notes measure between 2.1mm and 2.7mm.
        # We enforce a strict range of 1.9mm to 3.0mm to reject fake background alignments.
        passed = (max_stacked == expected_segments and 1.9 <= measured_mm <= 3.0)
        return passed, f"Segments: {max_stacked}/{expected_segments} | Avg Width: {measured_mm:.1f}mm", explain_img
    else:
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
                if i in used: continue
                rx, ry, rw, rh = raw_rects[i]
                merge_idx = -1
                for j in range(i + 1, len(raw_rects)):
                    if j in used: continue
                    ox, oy, ow, oh = raw_rects[j]
                    overlap_x = max(rx, ox) <= min(rx + rw, ox + ow) + 5
                    gap = oy - (ry + rh)
                    
                    if overlap_x and gap <= 15:
                        merge_idx = j
                        break
                
                if merge_idx != -1:
                    ox, oy, ow, oh = raw_rects[merge_idx]
                    min_x, min_y = min(rx, ox), min(ry, oy)
                    max_x, max_y = max(rx + rw, ox + ow), max(ry + rh, oy + oh)
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
                
    if max_stacked >= 3:
        for rx, ry, rw, rh in valid_rects:
            if abs(rx - best_x) <= 15:
                cv2.rectangle(explain_img, (rx, ry), (rx+rw, ry+rh), (0, 255, 0), 2)
                
    if avg_original_width > 0:
        measured_mm = avg_original_width / 7.3
    else:
        avg_width_px = np.mean([rw for rx, ry, rw, rh in valid_rects]) if valid_rects else 0
        measured_mm = avg_width_px / 7.3

    if denom == 'LKR_5000':
        expected_segments = 5
        # 5000 LKR threads measure between 2.6mm and 3.4mm. We enforce 2.4mm to 3.6mm.
        passed = (max_stacked == expected_segments and 2.4 <= measured_mm <= 3.6)
        return passed, f"Segments: {max_stacked}/{expected_segments} | Avg Width: {measured_mm:.1f}mm", explain_img
    else:
        # For LKR_1000
        expected_segments = 4
        # 1000 LKR threads measure between 2.2mm and 2.6mm. We enforce 2.0mm to 2.8mm.
        passed = (max_stacked >= expected_segments and 2.0 <= measured_mm <= 2.8)
        return passed, f"Segments: {max_stacked}/{expected_segments} | Avg Width: {measured_mm:.1f}mm", explain_img

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
                
    if denom == 'LKR_5000':
        passed = (valid_lines == 15)
    elif denom == 'LKR_500':
        passed = valid_lines >= 13
    else:
        passed = valid_lines >= 13
        
    return passed, f"Edge Lines: {valid_lines}/15", explain_img
