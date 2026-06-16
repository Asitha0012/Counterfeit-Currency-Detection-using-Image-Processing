import os
import glob
import sys
import time
import numpy as np
import matplotlib.pyplot as plt

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(base_dir, 'src'))

import cv2
from align_note import align_note
from evaluate_lkr import (
    verify_visual_feature, verify_blind_dots, 
    verify_asymmetric_serial, verify_vertical_red_serial,
    verify_security_thread, verify_edge_lines
)

def run_latency_analysis():
    augmented_base = os.path.join(base_dir, "data", "augmented_testing")
    img_dir = os.path.join(augmented_base, "Genuine", "LKR_1000")
    img_paths = glob.glob(os.path.join(img_dir, '*.jpg'))[:100]
    
    if not img_paths:
        print("No images found for latency testing.")
        return
        
    print(f"Running latency analysis on {len(img_paths)} frames...")
    
    phase1_times = []
    phase2_times = []
    phase3_times = []
    
    for path in img_paths:
        raw_img = cv2.imread(path)
        if raw_img is None: continue
        
        # --- PHASE 1: ALIGNMENT ---
        t0 = time.time()
        aligned_img = align_note(raw_img, "LKR_1000")
        t1 = time.time()
        phase1_times.append(t1 - t0)
        
        # --- PHASE 2: VETO GATES (F1, F7, F11) ---
        t2 = time.time()
        verify_visual_feature(aligned_img, "LKR_1000", 1) # F1
        verify_visual_feature(aligned_img, "LKR_1000", 7) # F7
        verify_security_thread(aligned_img, "LKR_1000") # F11
        t3 = time.time()
        phase2_times.append(t3 - t2)
        
        # --- PHASE 3: FLAT VOTING (Remaining 9) ---
        t4 = time.time()
        for f_id in [2, 3, 4, 5, 6]:
            verify_visual_feature(aligned_img, "LKR_1000", f_id)
        verify_blind_dots(aligned_img, "LKR_1000")
        verify_asymmetric_serial(aligned_img, "LKR_1000")
        verify_vertical_red_serial(aligned_img, "LKR_1000")
        verify_edge_lines(aligned_img, "LKR_1000")
        t5 = time.time()
        phase3_times.append(t5 - t4)
        
    avg_p1 = np.mean(phase1_times)
    avg_p2 = np.mean(phase2_times)
    avg_p3 = np.mean(phase3_times)
    total_avg = avg_p1 + avg_p2 + avg_p3
    
    print("\n" + "="*50)
    print("AVERAGE ALGORITHMIC PROCESSING LATENCY PER FRAME")
    print("="*50)
    print(f"Phase 1 (Alignment/ORB/Pre-process) : {avg_p1:.3f} s")
    print(f"Phase 2 (Stage A Veto Gates)      : {avg_p2:.3f} s")
    print(f"Phase 3 (Stage B Flat Voting)     : {avg_p3:.3f} s")
    print("-" * 50)
    print(f"Total Execution                   : {total_avg:.3f} s")
    
    # Save a bar chart for the paper
    phases = ['Phase 1\n(Alignment)', 'Phase 2\n(Veto Gates)', 'Phase 3\n(Flat Voting)']
    times = [avg_p1, avg_p2, avg_p3]
    
    plt.figure(figsize=(8, 5))
    bars = plt.bar(phases, times, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    plt.ylabel('Average Latency (Seconds)')
    plt.title(f'Computational Latency Analysis (Total: {total_avg:.2f}s)')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.05, f"{yval:.2f}s", ha='center', va='bottom', fontweight='bold')
        
    brain_dir = os.path.join(os.environ['USERPROFILE'], '.gemini', 'antigravity-ide', 'brain', '8912d3ae-1929-438e-812f-4925e7c66e2a')
    out_path = os.path.join(brain_dir, 'Latency_Analysis.png')
    plt.savefig(out_path)
    print(f"\nLatency chart saved to {out_path}")

if __name__ == "__main__":
    run_latency_analysis()
