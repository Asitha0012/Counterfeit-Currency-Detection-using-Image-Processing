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
    
    for thresh_value in range(90, 155, 5):
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
            gray = cv2.cvtColor(center_crop, cv2.COLOR_BGR2GRAY)
            mask = (gray > 15) & (gray < 252)
            if np.sum(mask) == 0:
                return False
                
            h_vals = hsv[:, :, 0][mask]
            s_vals = hsv[:, :, 1][mask]
            
            h_rad = (h_vals.astype(np.float32) * 2.0) * (np.pi / 180.0)
            avg_x = np.mean(np.cos(h_rad))
            avg_y = np.mean(np.sin(h_rad))
            avg_hue_deg = np.arctan2(avg_y, avg_x) * (180.0 / np.pi)
            if avg_hue_deg < 0:
                avg_hue_deg += 360.0
            avg_hue_opencv = avg_hue_deg / 2.0
            avg_sat = np.mean(s_vals)
            
            # Base color range for 500 (stone grey/greenish-yellow)
            base_color_passed = (20.0 <= avg_hue_opencv <= 75.0) and (5.0 <= avg_sat <= 50.0)
            
            # Thread check: search narrow coordinate band for green shift pixels
            thread_crop = img[100:400, 560:630]
            max_ratio = 0.0
            for col in range(thread_crop.shape[1] - 15):
                strip = thread_crop[:, col:col+15]
                gray_strip = cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY)
                shift_mask = (strip[:, :, 1].astype(np.int16) > strip[:, :, 2].astype(np.int16) + 3) & (gray_strip < 210)
                ratio = np.sum(shift_mask) / shift_mask.size
                if ratio > max_ratio:
                    max_ratio = ratio
            thread_passed = (max_ratio > 0.01)
            
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
            gray = cv2.cvtColor(center_crop, cv2.COLOR_BGR2GRAY)
            mask = (gray > 15) & (gray < 252)
            if np.sum(mask) == 0:
                return False
                
            h_vals = hsv[:, :, 0][mask]
            s_vals = hsv[:, :, 1][mask]
            
            h_rad = (h_vals.astype(np.float32) * 2.0) * (np.pi / 180.0)
            avg_x = np.mean(np.cos(h_rad))
            avg_y = np.mean(np.sin(h_rad))
            avg_hue_deg = np.arctan2(avg_y, avg_x) * (180.0 / np.pi)
            if avg_hue_deg < 0:
                avg_hue_deg += 360.0
            avg_hue_opencv = avg_hue_deg / 2.0
            avg_sat = np.mean(s_vals)
            
            # Base color range for 2000 (pink/magenta)
            base_color_passed = (135.0 <= avg_hue_opencv <= 175.0) and (10.0 <= avg_sat <= 90.0)
            
            # Thread check for 2000 (intensity change valley check)
            crop_thread = cv2.cvtColor(img[100:350, 560:630], cv2.COLOR_BGR2GRAY)
            col_means = np.mean(crop_thread, axis=0)
            valley_depth = np.mean(col_means) - np.min(col_means)
            thread_passed = (valley_depth >= 14.0)
            
        return base_color_passed and thread_passed

def verify_watermark(img, denom):
    crop = img[150:350, 880:1080] if denom == '500' else img[120:300, 850:1050]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    std_dev = np.std(gray)
    edges = cv2.Canny(gray, 30, 100)
    edge_density = np.sum(edges > 0) / edges.size
    return (std_dev >= 5.0) and (std_dev <= 70.0) and (edge_density > 0.001) and (edge_density < 0.25)

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
    verdict = (passed_features >= 10)
    return verdict, passed_features, feature_statuses

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
    print("=" * 70)
    
    # Real-World dataset (n=31): 19 Genuine, 12 Real Counterfeits (non-simulated)
    real_y_true = []
    real_y_pred = []
    real_times = []
    
    # Robustness probe (n=38): 38 Simulated Counterfeits
    sim_y_true = []
    sim_y_pred = []
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
            if not file.endswith(".jpg"):
                continue
            path = os.path.join(folder, file)
            
            start_note = time.perf_counter()
            verdict, passed, statuses = analyze_note(path, denom)
            end_note = time.perf_counter()
            
            elapsed = (end_note - start_note) * 1000.0 # ms
            pred_label = 1 if verdict else 0
            
            # Determine if this is a simulated note or a real note
            is_sim = "sim_" in file or "synthetic" in folder
            
            if label == 1: # Genuine is always real
                real_y_true.append(1)
                real_y_pred.append(pred_label)
                real_times.append(elapsed)
                real_gen_total += 1
                for f_idx, s in enumerate(statuses):
                    if s: real_feature_stats[f_idx][0] += 1
            else: # Counterfeit
                if is_sim:
                    sim_y_true.append(0)
                    sim_y_pred.append(pred_label)
                    sim_times.append(elapsed)
                else: # Real Counterfeit
                    real_y_true.append(0)
                    real_y_pred.append(pred_label)
                    real_times.append(elapsed)
                    real_fake_total += 1
                    for f_idx, s in enumerate(statuses):
                        if s: real_feature_stats[f_idx][1] += 1
            
            category_str = "Genuine" if label == 1 else ("Counterfeit (Simulated)" if is_sim else "Counterfeit (Real)")
            print(f"  File: {file:<25} Passed: {passed:>2}/12 ({passed/12.0*100.0:4.1f}%)  Time: {elapsed:5.1f} ms  Verdict: {'Genuine' if verdict else 'Counterfeit':<11} {'[CORRECT]' if pred_label == label else '[ERROR]'}")
            
    # Calculate Real-World (n=31) metrics
    real_y_true = np.array(real_y_true)
    real_y_pred = np.array(real_y_pred)
    real_tp = np.sum((real_y_true == 1) & (real_y_pred == 1))
    real_tn = np.sum((real_y_true == 0) & (real_y_pred == 0))
    real_fp = np.sum((real_y_true == 0) & (real_y_pred == 1))
    real_fn = np.sum((real_y_true == 1) & (real_y_pred == 0))
    real_total = len(real_y_true)
    
    real_accuracy, real_ci_l, real_ci_u = report_accuracy_with_confidence(real_tp + real_tn, real_total)
    real_precision = real_tp / (real_tp + real_fp) if (real_tp + real_fp) > 0 else 0
    real_recall = real_tp / (real_tp + real_fn) if (real_tp + real_fn) > 0 else 0
    real_f1 = 2 * real_precision * real_recall / (real_precision + real_recall) if (real_precision + real_recall) > 0 else 0
    real_avg_time = sum(real_times) / len(real_times) if real_times else 0
    
    # Calculate Synthetic Robustness (n=38) metrics
    sim_y_true = np.array(sim_y_true)
    sim_y_pred = np.array(sim_y_pred)
    sim_total = len(sim_y_true)
    sim_tn = np.sum((sim_y_true == 0) & (sim_y_pred == 0))
    sim_fp = np.sum((sim_y_true == 0) & (sim_y_pred == 1))
    sim_accuracy = sim_tn / sim_total if sim_total > 0 else 0
    sim_avg_time = sum(sim_times) / len(sim_times) if sim_times else 0
    
    print("\n" + "=" * 70)
    print("EVALUATION 1: PRIMARY BENCHMARK ON REAL BANKNOTES (n=31)")
    print("=" * 70)
    print(f"Total Banknotes Analysed     : {real_total}")
    print(f"Classification Accuracy      : {real_accuracy * 100:.2f}%")
    print(f"95% Confidence Interval (CI) : [{real_ci_l*100:.1f}%, {real_ci_u*100:.1f}%] (Wilson Score)")
    print(f"Classifier Precision         : {real_precision * 100:.2f}%")
    print(f"Classifier Recall (Sens.)    : {real_recall * 100:.2f}%")
    print(f"F1 Classifier Score          : {real_f1 * 100:.2f}%")
    print(f"Average Processing Speed     : {real_avg_time:.1f} ms per banknote")
    print("\n----------------------------------------------------------------------")
    print("CONFUSION MATRIX (REAL BANKNOTES)")
    print("----------------------------------------------------------------------")
    print(f"                     Predicted Genuine    Predicted Counterfeit")
    print(f"Actual Genuine       {real_tp:<20} {real_fn:<20}")
    print(f"Actual Counterfeit   {real_fp:<20} {real_tn:<20}")
    
    print("\n" + "=" * 70)
    print("EVALUATION 2: ROBUSTNESS PROBE ON SYNTHESIZED BANKNOTES (n=38)")
    print("=" * 70)
    print(f"Total Banknotes Analysed     : {sim_total}")
    print(f"Attack Rejection Accuracy    : {sim_accuracy * 100:.2f}%")
    print(f"Average Processing Speed     : {sim_avg_time:.1f} ms per banknote")
    print("\n----------------------------------------------------------------------")
    print("DETECTION PROFILE (SYNTHETIC ATTACKS)")
    print("----------------------------------------------------------------------")
    print(f"Synthesized Counterfeits Correctly Rejected: {sim_tn} / {sim_total}")
    print(f"Synthesized Counterfeits Leaked (False Pos): {sim_fp} / {sim_total}")
    
    print("\n" + "=" * 70)
    print("REAL-WORLD FEATURE DISCRIMINATIVE EFFECTIVENESS ANALYSIS")
    print("=" * 70)
    print(f"Feature Number    Genuine Pass Rate (%)    Counterfeit Pass Rate (%)   Discriminability (Gen - Fake)")
    print("-" * 70)
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
    print("=" * 70)

if __name__ == '__main__':
    run_evaluation()
