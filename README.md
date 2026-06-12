<div align="center">

# 🕵️‍♂️ Counterfeit Currency Detection (LKR)
**A Fully Explainable, Classical Computer Vision Pipeline for Sri Lankan Rupee Authentication**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![OpenCV](https://img.shields.io/badge/opencv-%23white.svg?logo=opencv&logoColor=white)](https://opencv.org/)
[![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?logo=numpy&logoColor=white)](https://numpy.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

</div>

---

## 📌 Project Overview
This repository contains a highly-optimized, modular counterfeit currency detection system specifically designed for **Sri Lankan Rupee (LKR 1000 & 5000)** banknotes. The software utilizes Classical Computer Vision techniques to automatically align and mathematically verify **12 distinct security features** embedded by the Central Bank of Sri Lanka (CBSL).

Unlike "Black Box" deep learning models, this system is **100% Explainable**, entirely deterministic, and operates with a robust Two-Stage Hybrid Classifier featuring strict security Veto Gates. 

---

## 🚀 The 4-Stage Classification Pipeline

1. **Stage 1: Automatic Alignment (Pre-processing)**
   - Uses **Canny Edge Detection** to segment the note from cluttered backgrounds.
   - Automatically straightens and warps the image to standard template dimensions using **ORB Homography** and Normalized Cross-Correlation (NCC).

2. **Stage 2: Visual Features (SSIM)**
   - Employs **Template Matching** for coarse localization.
   - Uses **Structural Similarity Index Measure (SSIM)** to mathematically verify the ink structure of visual elements (e.g., Lion Emblem, Butterfly Motif).

3. **Stage 3: Programmatic Features (Geometric Analysis)**
   - **Blind Recognition Dots:** Isolated using Adaptive Otsu's Binarization & Contour Aspect-Ratio Filtering.
   - **Starchrome Security Thread:** Verified using Morphological Opening.
   - **Vertical Red Serial Number:** Isolated using HSV Color Masking.

4. **Stage 4: Hybrid Classification (The Verdict)**
   - **The Veto Gate:** Features physically unforgeable by consumer hardware (Micro-printing, Watermark, Security Thread) must pass perfectly. If any fail, the system acts as a **Fail-Safe** and immediately blocks the note.
   - **Flat Voting:** If the Veto Gate passes, the system applies a ≥ 75% overall pass rate to declare the note Genuine.

---

## 📂 Repository Layout

```text
├── main_gui_lkr.py               # Main Application entry point (Tkinter GUI)
├── requirements.txt              # Python dependencies
├── src/                          # Core algorithmic engines
│   ├── align_note.py             # ORB Homography and perspective warping
│   └── evaluate_lkr.py           # Master evaluation script with all 12 feature logic gates
├── scripts/                      # Scientific testing automation
│   ├── augment_testing_data.py   # Synthesizes datasets using real-world lighting noise
│   └── test_augmented_datasets.py# Headless batch tester and Accuracy Reporter
├── data/                         # Structured image repository
│   ├── genuine/                  # High-res physical scans
│   ├── counterfeit/              # Physical counterfeit scans
│   └── templates/                # Mathematically cropped feature templates
└── docs/                         # Technical documentation and presentations
```

---

## ⚙️ Quick Start Guide

### 1. Install Dependencies
Install all required Python packages (No GPU required):
```bash
pip install -r requirements.txt
```

### 2. Launch the Desktop GUI
To run the standard user interface and test a note:
```bash
python main_gui_lkr.py
```
1. Click "Select Image" and upload a scanned or photographed LKR note.
2. The system will auto-align the note, run all 12 mathematical checks, and present an explainable bounding-box breakdown showing exactly why the note passed or failed.

### 3. Run the Automated Scientific Testing
To automatically generate a robust 400-image dataset and run the quantitative accuracy tests:
```bash
python scripts\augment_testing_data.py
python scripts\test_augmented_datasets.py
```
This will evaluate all notes and print the final Accuracy, Precision, and False-Positive metrics directly to your terminal.

---
<div align="center">
<i>Built for Academic Evaluation in Computer Vision</i>
</div>
