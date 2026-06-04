import cv2
import numpy as np
import os
import time
from skimage.metrics import structural_similarity as ssim

def calculate_ssim(template_img, query_img):
    if template_img is None or query_img is None or template_img.size == 0 or query_img.size == 0:
        return 0.0
    min_w = min(template_img.shape[1], query_img.shape[1])
    min_h = min(template_img.shape[0], query_img.shape[0])
    if min_w <= 0 or min_h <= 0:
        return 0.0
    img1 = cv2.resize(template_img, (min_w, min_h))
    img2 = cv2.resize(query_img, (min_w, min_h))
    if len(img1.shape) == 3:
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    if len(img2.shape) == 3:
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    return ssim(img1, img2)

def compute_orb(template_img, query_img):
    orb = cv2.ORB_create(700, 1.2, 8, 15)
    kpts1, descs1 = orb.detectAndCompute(template_img, None)
    kpts2, descs2 = orb.detectAndCompute(query_img, None)
    if descs1 is None or descs2 is None:
        return None, None
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descs1, descs2)
    dmatches = sorted(matches, key=lambda x: x.distance)
    if len(dmatches) < 6:
        return None, None
    src_pts = np.float32([kpts1[m.queryIdx].pt for m in dmatches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kpts2[m.trainIdx].pt for m in dmatches]).reshape(-1, 1, 2)
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    MIN_INLIERS = 6
    if mask is None or int(mask.sum()) < MIN_INLIERS:
        return None, None
    h, w = template_img.shape[:2]
    pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
    if M is not None:
        dst = cv2.perspectiveTransform(pts, M)
    else:
        dst = None
    return dst, dst_pts

def check_image_quality(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    mean_bright = np.mean(gray)
    is_blurry = lap_var < 100.0
    is_underexposed = mean_bright < 50.0
    is_overexposed = mean_bright > 230.0
    return not (is_blurry or is_underexposed or is_overexposed)

def verify_bleed_lines(img, side, denom):
    if denom == '500':
        crop = img[120:240, 12:35] if side == 'left' else img[120:260, 1135:1155]
        expected_range = (4.7, 5.6)
    else:
        crop = img[80:230, 10:30] if side == 'left' else img[90:230, 1140:1160]
        expected_range = (6.7, 7.6)
        
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 130, 255, cv2.THRESH_BINARY)
    width = thresh.shape[1]
    
    result = []
    num_of_cols = 0
    for j in range(width):
        col = thresh[:, j:j+1]
        count = 0
        for i in range(len(col)-1):
            p1 = 255 if col[i][0] not in [0, 255] else col[i][0]
            p2 = 255 if col[i+1][0] not in [0, 255] else col[i+1][0]
            if p1 == 255 and p2 == 0:
                count += 1
        if count > 0 and count < 10:
            result.append(count)
            num_of_cols += 1
            
    if num_of_cols != 0:
        avg_count = sum(result) / num_of_cols
    else:
        avg_count = -1
        
    status = (avg_count >= expected_range[0] and avg_count <= expected_range[1])
    return status, avg_count

def verify_number_panel(img, denom):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if denom == '500':
        crop = gray[410:500, 700:1080]
    else:
        crop = gray[360:440, 760:1080]
        
    h_img, w_img = crop.shape[:2]
    
    if denom == '500':
        start_thresh = 95
    else:
        start_thresh = 90
        
    for thresh_value in range(start_thresh, 155, 5):
        _, thresh = cv2.threshold(crop, thresh_value, 255, cv2.THRESH_BINARY)
        img_masked = cv2.bitwise_and(crop, crop, mask=thresh)
        contours, _ = cv2.findContours(img_masked, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        
        bounding_rect_list = []
        for contour in contours:
            [x, y, w, h] = cv2.boundingRect(contour)
            if x != 0:
                bounding_rect_list.append([x, y, w, h])
        bounding_rect_list.sort()
        
        min_area = 150
        res_list = []
        for rect in bounding_rect_list:
            if rect[2] * rect[3] > min_area:
                res_list.append(rect)
                
        i = 0
        while i < len(res_list):
            [x, y, w, h] = res_list[i]
            j = i + 1
            while j < len(res_list):
                [x0, y0, w0, h0] = res_list[j]
                if (x + w) >= x0 + w0:
                    res_list.pop(j)
                else:
                    break
            i += 1
            
        i = 0
        while i < len(res_list):
            [x, y, w, h] = res_list[i]
            if (h_img - (y + h)) > 40:
                res_list.pop(i)
            elif h < 17:
                res_list.pop(i)
            else:
                i += 1
                
        if len(res_list) == 9:
            return True
            
    return False

def verify_color_profile(img, denom):
    # Purely image-processing based verification
    is_gray = np.all(img[:, :, 0] == img[:, :, 1]) and np.all(img[:, :, 0] == img[:, :, 2])
    
    if denom == '500':
        if is_gray:
            crop = cv2.cvtColor(img[100:400, 560:630], cv2.COLOR_BGR2GRAY)
            col_means = np.mean(crop, axis=0)
            valley_depth = np.mean(col_means) - np.min(col_means)
            thread_passed = (valley_depth >= 14.0)
            base_color_passed = True
        else:
            center_crop = img[100:400, 200:900]
            hsv = cv2.cvtColor(center_crop, cv2.COLOR_BGR2HSV)
            lower_gray = np.array([10, 2, 40])
            upper_gray = np.array([70, 70, 240])
            mask = cv2.inRange(hsv, lower_gray, upper_gray)
            ratio = np.sum(mask > 0) / mask.size
            base_color_passed = ratio > 0.4
            
            thread_crop = img[100:400, 580:610]
            g_b_mask = (thread_crop[:, :, 1].astype(np.int16) > thread_crop[:, :, 2].astype(np.int16) + 8) | \
                       (thread_crop[:, :, 0].astype(np.int16) > thread_crop[:, :, 2].astype(np.int16) + 8)
            thread_ratio = np.sum(g_b_mask) / g_b_mask.size
            thread_passed = thread_ratio > 0.005
            
        return base_color_passed and thread_passed
    else:
        # denom == '2000'
        if is_gray:
            crop = cv2.cvtColor(img[100:350, 560:630], cv2.COLOR_BGR2GRAY)
            col_means = np.mean(crop, axis=0)
            valley_depth = np.mean(col_means) - np.min(col_means)
            thread_passed = (valley_depth >= 14.0)
            base_color_passed = True
        else:
            center_crop = img[100:350, 200:900]
            hsv = cv2.cvtColor(center_crop, cv2.COLOR_BGR2HSV)
            lower_pink = np.array([130, 20, 40])
            upper_pink = np.array([178, 255, 255])
            mask = cv2.inRange(hsv, lower_pink, upper_pink)
            ratio = np.sum(mask > 0) / mask.size
            base_color_passed = ratio > 0.1
            
            thread_crop = img[100:350, 580:610]
            g_b_mask = (thread_crop[:, :, 1].astype(np.int16) > thread_crop[:, :, 2].astype(np.int16) + 8) | \
                       (thread_crop[:, :, 0].astype(np.int16) > thread_crop[:, :, 2].astype(np.int16) + 8)
            thread_ratio = np.sum(g_b_mask) / g_b_mask.size
            thread_passed = thread_ratio > 0.005
            
        return base_color_passed and thread_passed

def verify_watermark(img, denom):
    crop = img[150:350, 880:1080] if denom == '500' else img[120:300, 850:1050]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    std_dev = np.std(gray)
    edges = cv2.Canny(gray, 30, 100)
    edge_density = np.sum(edges > 0) / edges.size
    return (std_dev >= 8.0) and (std_dev <= 60.0) and (edge_density > 0.005) and (edge_density < 0.15)

def analyze_note(image_path, denom):
    img = cv2.imread(image_path)
    if img is None:
        return False, 0, [False]*12
        
    # Preprocessing
    if denom == '500':
        img = cv2.resize(img, (1167, 519))
    else:
        img = cv2.resize(img, (1165, 455))
        
    blur_img = cv2.GaussianBlur(img, (5, 5), 0)
    gray_img = cv2.cvtColor(blur_img, cv2.COLOR_BGR2GRAY)
    
    # Apply Contrast Limited Adaptive Histogram Equalization (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray_img = clahe.apply(gray_img)
    
    feature_statuses = [False] * 12
    
    # 1. Quality validation
    quality_passed = check_image_quality(img)
    # Note: Quality check is an execution safeguard, features 1-12 form classification checks.
    
    # 2. Features 1 to 7 (ORB + SSIM template matching)
    search_areas = {
        '500': [
            [200,300,200,370], [1050,1500,300,450], [100,450,20,120],
            [690,1050,20,120], [820,1050,350,430], [700,810,330,430], [400,650,0,100]
        ],
        '2000': [
            [200,270,160,330], [1050,1500,250,400], [50,400,0,100],
            [750,1050,0,100], [850,1050,280,380], [700,820,290,370], [400,650,0,100]
        ]
    }
    
    feature_limits = {
        '500': [
            [12000,17000], [10000,18000], [20000,30000],
            [24000,36000], [15000,25000], [7000,13000], [11000,18000]
        ],
        '2000': [
            [10000,14000], [9000,15000], [17000,21500],
            [19000,28000], [17500,23000], [6500,9000], [10000,16000]
        ]
    }
    
    min_ssim_scores = {
        '500': [0.4, 0.4, 0.5, 0.4, 0.5, 0.45, 0.5],
        '2000': [0.45, 0.4, 0.45, 0.45, 0.5, 0.4, 0.5]
    }
    
    for j in range(7):
        search_area = search_areas[denom][j]
        mask_test = gray_img.copy()
        mask_test[:, :search_area[0]] = 0
        mask_test[:, search_area[1]:] = 0
        mask_test[:search_area[2], :] = 0
        mask_test[search_area[3]:, :] = 0
        
        feature_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "templates", denom, f"Feature {j+1}")
        templates = sorted([f for f in os.listdir(feature_dir) if f.endswith('.jpg')])
        
        max_score = -1
        score_set = []
        
        # Evaluate up to 6 templates for performance
        for t_file in templates[:6]:
            t_path = os.path.join(feature_dir, t_file)
            t_img = cv2.imread(t_path)
            if t_img is None:
                continue
            t_blur = cv2.GaussianBlur(t_img, (5, 5), 0)
            t_gray = cv2.cvtColor(t_blur, cv2.COLOR_BGR2GRAY)
            t_gray = clahe.apply(t_gray) # Apply CLAHE to template gray too
            
            dst, dst_pts = compute_orb(t_gray, mask_test)
            if dst is None:
                continue
                
            (x, y, w, h) = cv2.boundingRect(dst)
            min_area, max_area = feature_limits[denom][j]
            if w*h < min_area or w*h > max_area:
                (x, y, w, h) = cv2.boundingRect(dst_pts)
                if w*h < min_area or w*h > max_area:
                    continue
                    
            crop_img = blur_img[y:y+h, x:x+w]
            score = calculate_ssim(t_blur, crop_img)
            score_set.append(score)
            if score > max_score:
                max_score = score
                
        avg_score = sum(score_set) / len(score_set) if score_set else 0
        min_allowed = min_ssim_scores[denom][j]
        if avg_score >= min_allowed or max_score >= 0.79:
            feature_statuses[j] = True
            
    # 8. Left bleed lines
    lbl_passed, _ = verify_bleed_lines(img, 'left', denom)
    feature_statuses[7] = lbl_passed
        
    # 9. Right bleed lines
    rbl_passed, _ = verify_bleed_lines(img, 'right', denom)
    feature_statuses[8] = rbl_passed
        
    # 10. Number panel character count
    np_passed = verify_number_panel(img, denom)
    feature_statuses[9] = np_passed
        
    # 11. Color profile check
    color_passed = verify_color_profile(img, denom)
    feature_statuses[10] = color_passed
        
    # 12. Watermark check
    watermark_passed = verify_watermark(img, denom)
    feature_statuses[11] = watermark_passed
        
    passed_features = sum(feature_statuses)
    # Classification Verdict: A banknote is Genuine if it passes at least 10 features
    flat_verdict = (passed_features >= 10)
    # Veto-based Verdict: Also requires Feature 11 (Color & Thread Check, index 10) to pass
    veto_verdict = flat_verdict and feature_statuses[10]
    return flat_verdict, veto_verdict, passed_features, feature_statuses

def report_accuracy_with_confidence(correct, total, confidence=0.95):
    """Wilson Score Interval implementation for accurate confidence reporting on small sample sizes."""
    if total == 0:
        return 0, 0, 0
    accuracy = correct / total
    z = 1.96 # 95% Confidence Interval z-score
    denominator = 1 + z**2 / total
    center = (accuracy + z**2 / (2*total)) / denominator
    margin = (z * np.sqrt(accuracy*(1-accuracy)/total + z**2/(4*total**2))) / denominator
    lower = center - margin
    upper = center + margin
    return accuracy, lower, upper
def run_evaluation():
    base_dir = os.path.dirname(os.path.dirname(__file__))
    categories = [
        (os.path.join(base_dir, "data", "genuine", "500"), "500", 1),      # Genuine
        (os.path.join(base_dir, "data", "genuine", "2000"), "2000", 1),    # Genuine
        (os.path.join(base_dir, "data", "counterfeit", "500"), "500", 0),   # Counterfeit (Real)
        (os.path.join(base_dir, "data", "counterfeit", "2000"), "2000", 0),  # Counterfeit (Real)
        (os.path.join(base_dir, "data", "synthetic", "500"), "500", 0),     # Counterfeit (Simulated)
        (os.path.join(base_dir, "data", "synthetic", "2000"), "2000", 0)    # Counterfeit (Simulated)
    ]
    
    print("Executing Quantitative System Evaluation (Dual-Evaluation Headless Pipeline)...")
    print("=" * 75)
    
    # Real-World dataset (n=31): 19 Genuine, 12 Real Counterfeits (non-simulated)
    real_y_true = []
    real_y_pred_flat = []
    real_y_pred_veto = []
    real_times = []
    
    # Robustness probe (n=38): 38 Simulated Counterfeits
    sim_y_true = []
    sim_y_pred_flat = []
    sim_y_pred_veto = []
    sim_times = []
    
    # Track overall features stats for real-world notes
    real_feature_stats = [[0, 0] for _ in range(12)]
    real_gen_total = 0
    real_fake_total = 0
    
    for folder, denom, label in categories:
        if not os.path.exists(folder):
            print(f"Directory not found: {folder}")
            continue
            
        print(f"\nProcessing directory: {folder} ({'Genuine' if label == 1 else 'Counterfeit'})")
        for file in sorted(os.listdir(folder)):
            if not file.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff")):
                continue
            path = os.path.join(folder, file)
            
            start_note = time.perf_counter()
            flat_verdict, veto_verdict, passed, statuses = analyze_note(path, denom)
            end_note = time.perf_counter()
            
            elapsed = (end_note - start_note) * 1000.0 # ms
            pred_flat = 1 if flat_verdict else 0
            pred_veto = 1 if veto_verdict else 0
            
            # Determine if this is a simulated note or a real note
            is_sim = "sim_" in file or "synthetic" in folder
            
            if label == 1: # Genuine is always real
                real_y_true.append(1)
                real_y_pred_flat.append(pred_flat)
                real_y_pred_veto.append(pred_veto)
                real_times.append(elapsed)
                real_gen_total += 1
                for f_idx, s in enumerate(statuses):
                    if s: real_feature_stats[f_idx][0] += 1
            else: # Counterfeit
                if is_sim:
                    sim_y_true.append(0)
                    sim_y_pred_flat.append(pred_flat)
                    sim_y_pred_veto.append(pred_veto)
                    sim_times.append(elapsed)
                else: # Real Counterfeit
                    real_y_true.append(0)
                    real_y_pred_flat.append(pred_flat)
                    real_y_pred_veto.append(pred_veto)
                    real_times.append(elapsed)
                    real_fake_total += 1
                    for f_idx, s in enumerate(statuses):
                        if s: real_feature_stats[f_idx][1] += 1
            
            category_str = "Genuine" if label == 1 else ("Counterfeit (Simulated)" if is_sim else "Counterfeit (Real)")
            print(f"  File: {file:<25} Passed: {passed:>2}/12 ({passed/12.0*100.0:4.1f}%)  Time: {elapsed:5.1f} ms  Flat Verdict: {'Genuine' if flat_verdict else 'Counterfeit':<11}  Veto Verdict: {'Genuine' if veto_verdict else 'Counterfeit':<11}")
            
    # Calculate Real-World (n=31) metrics
    real_y_true = np.array(real_y_true)
    real_y_pred_flat = np.array(real_y_pred_flat)
    real_y_pred_veto = np.array(real_y_pred_veto)
    real_total = len(real_y_true)
    
    # Flat metrics
    flat_tp = np.sum((real_y_true == 1) & (real_y_pred_flat == 1))
    flat_tn = np.sum((real_y_true == 0) & (real_y_pred_flat == 0))
    flat_fp = np.sum((real_y_true == 0) & (real_y_pred_flat == 1))
    flat_fn = np.sum((real_y_true == 1) & (real_y_pred_flat == 0))
    flat_accuracy, flat_ci_l, flat_ci_u = report_accuracy_with_confidence(flat_tp + flat_tn, real_total)
    flat_precision = flat_tp / (flat_tp + flat_fp) if (flat_tp + flat_fp) > 0 else 0
    flat_recall = flat_tp / (flat_tp + flat_fn) if (flat_tp + flat_fn) > 0 else 0
    flat_f1 = 2 * flat_precision * flat_recall / (flat_precision + flat_recall) if (flat_precision + flat_recall) > 0 else 0
    
    # Veto metrics
    veto_tp = np.sum((real_y_true == 1) & (real_y_pred_veto == 1))
    veto_tn = np.sum((real_y_true == 0) & (real_y_pred_veto == 0))
    veto_fp = np.sum((real_y_true == 0) & (real_y_pred_veto == 1))
    veto_fn = np.sum((real_y_true == 1) & (real_y_pred_veto == 0))
    veto_accuracy, veto_ci_l, veto_ci_u = report_accuracy_with_confidence(veto_tp + veto_tn, real_total)
    veto_precision = veto_tp / (veto_tp + veto_fp) if (veto_tp + veto_fp) > 0 else 0
    veto_recall = veto_tp / (veto_tp + veto_fn) if (veto_tp + veto_fn) > 0 else 0
    veto_f1 = 2 * veto_precision * veto_recall / (veto_precision + veto_recall) if (veto_precision + veto_recall) > 0 else 0
    
    real_avg_time = sum(real_times) / len(real_times) if real_times else 0
    
    # Calculate Synthetic Robustness (n=38) metrics
    sim_y_true = np.array(sim_y_true)
    sim_y_pred_flat = np.array(sim_y_pred_flat)
    sim_y_pred_veto = np.array(sim_y_pred_veto)
    sim_total = len(sim_y_true)
    
    sim_flat_tn = np.sum((sim_y_true == 0) & (sim_y_pred_flat == 0))
    sim_flat_fp = np.sum((sim_y_true == 0) & (sim_y_pred_flat == 1))
    sim_flat_accuracy = sim_flat_tn / sim_total if sim_total > 0 else 0
    
    sim_veto_tn = np.sum((sim_y_true == 0) & (sim_y_pred_veto == 0))
    sim_veto_fp = np.sum((sim_y_true == 0) & (sim_y_pred_veto == 1))
    sim_veto_accuracy = sim_veto_tn / sim_total if sim_total > 0 else 0
    
    sim_avg_time = sum(sim_times) / len(sim_times) if sim_times else 0
    
    print("\n" + "=" * 75)
    print("EVALUATION 1: PRIMARY BENCHMARK ON REAL BANKNOTES (n=31)")
    print("=" * 75)
    print(f"Total Banknotes Analysed     : {real_total}")
    print(f"Average Processing Speed     : {real_avg_time:.1f} ms per banknote")
    print("\n  [Flat Voting Classifier (Baseline)]")
    print(f"  Classification Accuracy      : {flat_accuracy * 100:.2f}%")
    print(f"  95% Confidence Interval (CI) : [{flat_ci_l*100:.1f}%, {flat_ci_u*100:.1f}%] (Wilson Score)")
    print(f"  Classifier Precision         : {flat_precision * 100:.2f}%")
    print(f"  Classifier Recall (Sens.)    : {flat_recall * 100:.2f}%")
    print(f"  F1 Classifier Score          : {flat_f1 * 100:.2f}%")
    print("  CONFUSION MATRIX:")
    print(f"                       Predicted Gen    Predicted Fake")
    print(f"  Actual Genuine       {flat_tp:<16} {flat_fn:<16}")
    print(f"  Actual Counterfeit   {flat_fp:<16} {flat_tn:<16}")
    
    print("\n  [Veto-Based Classifier (Feature 11 Override)]")
    print(f"  Classification Accuracy      : {veto_accuracy * 100:.2f}%")
    print(f"  95% Confidence Interval (CI) : [{veto_ci_l*100:.1f}%, {veto_ci_u*100:.1f}%] (Wilson Score)")
    print(f"  Classifier Precision         : {veto_precision * 100:.2f}%")
    print(f"  Classifier Recall (Sens.)    : {veto_recall * 100:.2f}%")
    print(f"  F1 Classifier Score          : {veto_f1 * 100:.2f}%")
    print("  CONFUSION MATRIX:")
    print(f"                       Predicted Gen    Predicted Fake")
    print(f"  Actual Genuine       {veto_tp:<16} {veto_fn:<16}")
    print(f"  Actual Counterfeit   {veto_fp:<16} {veto_tn:<16}")
    
    print("\n" + "=" * 75)
    print("EVALUATION 2: ROBUSTNESS PROBE ON SYNTHESIZED BANKNOTES (n=38)")
    print("=" * 75)
    print(f"Total Banknotes Analysed     : {sim_total}")
    print(f"Average Processing Speed     : {sim_avg_time:.1f} ms per banknote")
    print(f"\n  [Flat Voting Classifier (Baseline)]")
    print(f"  Attack Rejection Accuracy    : {sim_flat_accuracy * 100:.2f}%")
    print(f"  Counterfeits Rejected        : {sim_flat_tn} / {sim_total}")
    print(f"  Counterfeits Leaked          : {sim_flat_fp} / {sim_total}")
    
    print(f"\n  [Veto-Based Classifier (Feature 11 Override)]")
    print(f"  Attack Rejection Accuracy    : {sim_veto_accuracy * 100:.2f}%")
    print(f"  Counterfeits Rejected        : {sim_veto_tn} / {sim_total}")
    print(f"  Counterfeits Leaked          : {sim_veto_fp} / {sim_total}")
    
    print("\n" + "=" * 75)
    print("REAL-WORLD FEATURE DISCRIMINATIVE EFFECTIVENESS ANALYSIS")
    print("=" * 75)
    print(f"Feature Number    Genuine Pass Rate (%)    Counterfeit Pass Rate (%)   Discriminability (Gen - Fake)")
    print("-" * 75)
    feature_names = [
        "F1 (Template 1)   ",
        "F2 (Template 2)   ",
        "F3 (Template 3)   ",
        "F4 (Template 4)   ",
        "F5 (Template 5)   ",
        "F6 (Template 6)   ",
        "F7 (Template 7)   ",
        "F8 (Left Bleeds)  ",
        "F9 (Right Bleeds) ",
        "F10 (Num Panel)   ",
        "F11 (Color HSV)   ",
        "F12 (Watermark)   "
    ]
    for idx, stats in enumerate(real_feature_stats):
        gen_pass_pct = (stats[0] / real_gen_total * 100) if real_gen_total > 0 else 0
        fake_pass_pct = (stats[1] / real_fake_total * 100) if real_fake_total > 0 else 0
        discriminability = gen_pass_pct - fake_pass_pct
        print(f"Feature {idx+1:<2} {feature_names[idx]} : {gen_pass_pct:5.1f}%                {fake_pass_pct:5.1f}%                      {discriminability:+5.1f}%")
    print("=" * 75)

if __name__ == '__main__':
    run_evaluation()
