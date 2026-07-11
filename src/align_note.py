"""
Automatic Note Alignment via NCC Coarse Matching and ORB-Based Homography

This module provides pre-alignment for banknote images before feature extraction.
It uses Normalized Cross Correlation (NCC) on a small thumbnail to:
1. Coarsely detect and correct cardinal rotations (0°, 90°, 180°, 270°).
2. Fine-align any minor rotation, perspective skew, or tilt using a RANSAC homography,
   but only if the distortion/tilt is significant (e.g. > 1.5° tilt) to avoid degrading
   perfectly aligned notes with homography interpolation noise.

If alignment fails (e.g., too few keypoints), it falls back to simple resizing
(the original behavior), ensuring the pipeline never crashes.
"""

import cv2
import numpy as np
import os

# Standard output dimensions for each denomination
STANDARD_DIMS = {
    '500': (1167, 519),   # (width, height)
    '2000': (1165, 455),
    'LKR_500': (1062, 500),
    'LKR_1000': (1104, 500),
    'LKR_5000': (1141, 500),
}

# Cache for loaded and preprocessed reference images to avoid reloading
_reference_cache = {}


def _get_reference(denom):
    """Load and cache the canonical reference note for a denomination."""
    if denom in _reference_cache:
        return _reference_cache[denom]
    
    # We must use the exact source images from which the feature templates were cropped
    # to ensure homography perfectly aligns the features to the templates.
    if denom == 'LKR_1000':
        source_img = '1000_G1.jpg'
    elif denom == 'LKR_5000':
        source_img = '5000_G1.jpg'
    elif denom == 'LKR_500':
        source_img = '500_G1.jpg'
    else:
        # Fallback
        source_img = '1000_G1.jpg' if denom == '500' else f'{denom}_s2.jpg'
    
    base_dir = os.path.dirname(os.path.dirname(__file__))
    ref_path = os.path.join(base_dir, 'data', 'genuine', denom, source_img)
    
    ref_img = cv2.imread(ref_path)
    if ref_img is None:
        return None
    
    # Resize reference to standard dimensions
    w, h = STANDARD_DIMS[denom]
    ref_img = cv2.resize(ref_img, (w, h))
    
    # Precompute grayscale and small thumbnail for coarse NCC matching
    ref_gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
    ref_small = cv2.resize(ref_gray, (200, 100))
    
    # Use ORB to find keypoints and descriptors for fine homography
    orb = cv2.ORB_create(5000, 1.2, 8, 15)
    kpts, descs = orb.detectAndCompute(ref_gray, None)
    
    _reference_cache[denom] = {
        'img': ref_img,
        'gray': ref_gray,
        'small': ref_small,
        'kpts': kpts,
        'descs': descs,
        'dims': (w, h),
    }
    return _reference_cache[denom]


def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - bl[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, M, (maxWidth, maxHeight))

def segment_note(image):
    """
    Detects and extracts a banknote from a cluttered background using edge detection
    and contour approximation. If no clear quadrilateral is found, returns original image.
    """
    ratio = image.shape[0] / 500.0
    orig = image.copy()
    resized = cv2.resize(image, (int(image.shape[1] / ratio), 500))

    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(gray, 75, 200)

    cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

    screenCnt = None
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            area = cv2.contourArea(approx)
            img_area = resized.shape[0] * resized.shape[1]
            # Ensure the detected contour takes up at least 20% of the image
            if area > 0.2 * img_area:
                screenCnt = approx
                break

    if screenCnt is None:
        return orig

    return four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)



def align_note(img, denom):
    """
    Automatically align a banknote image to standard orientation and dimensions.

    Uses Normalized Cross Correlation (NCC) to robustly detect the cardinal rotation
    (0°, 90°, 180°, 270°). If the note is tilted at an angle (e.g. 45°), it computes
    and applies a perspective warp via ORB homography.

    Parameters:
        img: Input BGR image (numpy array from cv2.imread)
        denom: Denomination string ('LKR_500', 'LKR_1000', 'LKR_5000')

    Returns:
        Aligned BGR image resized to standard dimensions
    """
    w, h = STANDARD_DIMS[denom]

    # 0. Attempt Background Segmentation
    img = segment_note(img)

    # Load the canonical reference
    ref_data = _get_reference(denom)
    if ref_data is None or ref_data['descs'] is None:
        # Reference not found — fall back to simple resize
        return cv2.resize(img, (w, h))

    # Coarse Orientation Correction via Normalized Cross Correlation
    ih, iw = img.shape[:2]

    # Pre-emptively correct portrait images to landscape (banknotes are always landscape)
    is_portrait = ih > iw * 1.2
    if is_portrait:
        img_landscape = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    else:
        img_landscape = img

    gray_landscape = cv2.cvtColor(img_landscape, cv2.COLOR_BGR2GRAY)

    # Resize to small thumbnail for extremely fast and robust NCC matching
    small_landscape = cv2.resize(gray_landscape, (200, 100))
    score0 = cv2.matchTemplate(small_landscape, ref_data['small'], cv2.TM_CCOEFF_NORMED)[0][0]

    small_landscape_180 = cv2.resize(cv2.rotate(gray_landscape, cv2.ROTATE_180), (200, 100))
    score180 = cv2.matchTemplate(small_landscape_180, ref_data['small'], cv2.TM_CCOEFF_NORMED)[0][0]

    if score180 > score0:
        coarse_img = cv2.rotate(img_landscape, cv2.ROTATE_180)
    else:
        coarse_img = img_landscape.copy()

    # Fine Orientation & Perspective Correction (Homography check)
    max_dim_fine = 1500
    ch, cw = coarse_img.shape[:2]
    scale_fine = max_dim_fine / max(ch, cw)

    if scale_fine < 1.0:
        coarse_scaled = cv2.resize(coarse_img, (int(cw * scale_fine), int(ch * scale_fine)))
    else:
        coarse_scaled = coarse_img.copy()
        scale_fine = 1.0

    coarse_gray = cv2.cvtColor(coarse_scaled, cv2.COLOR_BGR2GRAY)

    # Use 5000 features for extremely robust and accurate homography
    orb_fine = cv2.ORB_create(5000, 1.2, 8, 15)
    kpts_coarse, descs_coarse = orb_fine.detectAndCompute(coarse_gray, None)

    if descs_coarse is None or len(kpts_coarse) < 15:
        return cv2.resize(coarse_img, (w, h))

    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    try:
        raw_matches = bf.knnMatch(ref_data['descs'], descs_coarse, k=2)
    except cv2.error:
        return cv2.resize(coarse_img, (w, h))

    good_matches = []
    for pair in raw_matches:
        if len(pair) == 2:
            m, n = pair
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)

    if len(good_matches) < 15:
        return cv2.resize(coarse_img, (w, h))

    src_pts = np.float32([kpts_coarse[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([ref_data['kpts'][m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    M_scaled, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    if M_scaled is None:
        return cv2.resize(coarse_img, (w, h))

    theta = np.arctan2(M_scaled[1, 0], M_scaled[0, 0]) * 180 / np.pi

    if abs(theta) > 0.1:
        S = np.array([
            [scale_fine, 0, 0],
            [0, scale_fine, 0],
            [0, 0, 1]
        ])
        M_orig = M_scaled.dot(S)
        return cv2.warpPerspective(coarse_img, M_orig, (w, h), flags=cv2.INTER_LANCZOS4)
    else:
        return cv2.resize(coarse_img, (w, h))
