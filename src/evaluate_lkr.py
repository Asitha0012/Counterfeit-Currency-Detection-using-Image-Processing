import cv2
import numpy as np
import os

from align_note import align_note

# Import the newly modularized features
from features.visual_features import verify_visual_feature
from features.programmatic_features import (
    verify_blind_dots,
    verify_asymmetric_serial,
    verify_vertical_red_serial,
    verify_security_thread,
    verify_edge_lines
)

# =====================================================================
# MASTER PIPELINE EVALUATOR
# =====================================================================

def analyze_lkr_note(file_path, denom):
    """
    Executes the full Three-Algorithm Pipeline on an LKR Banknote.
    Returns: (final_verdict, message, score, feature_statuses, feature_images)
    """
    raw_img = cv2.imread(file_path)
    if raw_img is None:
        return False, False, "Could not read image", 0, [], []
        
    # STAGE 1: ALIGNMENT
    img = align_note(raw_img, denom)
    
    feature_statuses = []
    feature_images = []
    
    total_features = 0
    passed_features = 0
    veto_triggered = False
    
    # STAGE 2: VISUAL SSIM/ORB FEATURES (Algorithm 1)
    for f_id in range(1, 8):
        passed, msg, best_crop, score = verify_visual_feature(img, denom, f_id)
        if best_crop is None:
            best_crop = np.zeros((50, 50, 3), dtype=np.uint8)
        
        total_features += 1
        if passed: passed_features += 1
        
        feature_statuses.append((passed, msg))
        feature_images.append(best_crop)
        
        # Micro-Printing (F1) and Watermark (F7) are critical VETO gates (F11 is handled below)
        if (f_id in [1, 7]) and not passed:
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
