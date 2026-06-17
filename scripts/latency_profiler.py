import os
import sys
import time
import cv2

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(base_dir, 'src'))

from align_note import align_note
from features.visual_features import verify_visual_feature
from features.programmatic_features import (
    verify_blind_dots,
    verify_asymmetric_serial,
    verify_vertical_red_serial,
    verify_security_thread,
    verify_edge_lines
)

def profile_latency(img_path, denom):
    raw_img = cv2.imread(img_path)
    if raw_img is None:
        print("Failed to load image")
        return

    # Phase 1: Alignment
    start_time = time.perf_counter()
    img = align_note(raw_img, denom)
    phase1_time = time.perf_counter() - start_time

    # Phase 2: Veto Gates (F1, F7, F11)
    start_time = time.perf_counter()
    verify_visual_feature(img, denom, 1)
    verify_visual_feature(img, denom, 7)
    verify_security_thread(img, denom)
    phase2_time = time.perf_counter() - start_time

    # Phase 3: Flat Voting (Remaining 9 features)
    start_time = time.perf_counter()
    verify_visual_feature(img, denom, 2)
    verify_visual_feature(img, denom, 3)
    verify_visual_feature(img, denom, 4)
    verify_visual_feature(img, denom, 5)
    verify_visual_feature(img, denom, 6)
    verify_blind_dots(img, denom)
    verify_asymmetric_serial(img, denom)
    verify_vertical_red_serial(img, denom)
    verify_edge_lines(img, denom)
    phase3_time = time.perf_counter() - start_time

    total_time = phase1_time + phase2_time + phase3_time

    print("\n==================================================")
    print("COMPUTATIONAL LATENCY PROFILER")
    print("==================================================")
    print(f"Phase 1 (Alignment)            : {phase1_time:.4f} seconds")
    print(f"Phase 2 (Veto Gates F1, F7, F11): {phase2_time:.4f} seconds")
    print(f"Phase 3 (Flat Voting Features) : {phase3_time:.4f} seconds")
    print("--------------------------------------------------")
    print(f"Total Execution Time           : {total_time:.4f} seconds")
    
if __name__ == "__main__":
    img_path = os.path.join(base_dir, "data", "augmented_testing", "Genuine", "LKR_5000", "Scanned_20260608-1326-03_gen_1.jpg")
    if os.path.exists(img_path):
        profile_latency(img_path, "LKR_5000")
    else:
        print(f"Test image not found at {img_path}")
