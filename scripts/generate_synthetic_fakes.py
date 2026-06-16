import cv2
import numpy as np
import os
import glob
import random
import shutil

SEARCH_AREAS = {
    'LKR_1000': {
        1: (9, 355, 165, 494),      # Micro-Printing
        2: (13, 267, 243, 494),     # Butterfly
        3: (179, 319, 375, 474),    # Note Value
        4: (825, 83, 1097, 399),    # Bird
        5: (897, 1, 1044, 135),     # Lion Emblem
        6: (660, 394, 903, 491),    # Value in Text
        7: (186, 159, 308, 379)     # Watermark
    },
    'LKR_5000': {
        1: (5, 355, 157, 495),
        2: (8, 278, 223, 489),
        3: (157, 316, 354, 480),
        4: (861, 80, 1131, 407),
        5: (926, 4, 1074, 139),
        6: (686, 397, 924, 492),
        7: (190, 154, 307, 377)
    }
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

def apply_f1_blur(img, denom):
    x1, y1, x2, y2 = SEARCH_AREAS[denom][1]
    roi = img[y1:y2, x1:x2]
    img[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (31, 31), 15)
    return img

def apply_f2_eq(img, denom):
    x1, y1, x2, y2 = SEARCH_AREAS[denom][2]
    roi = img[y1:y2, x1:x2]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    hsv[:,:,2] = cv2.equalizeHist(hsv[:,:,2])
    img[y1:y2, x1:x2] = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return img

def apply_f3_shear(img, denom):
    x1, y1, x2, y2 = SEARCH_AREAS[denom][3]
    roi = img[y1:y2, x1:x2]
    pts1 = np.float32([[0,0], [roi.shape[1],0], [0,roi.shape[0]]])
    pts2 = np.float32([[0,roi.shape[0]*0.2], [roi.shape[1],0], [0,roi.shape[0]]])
    M = cv2.getAffineTransform(pts1,pts2)
    sheared = cv2.warpAffine(roi, M, (roi.shape[1], roi.shape[0]))
    img[y1:y2, x1:x2] = sheared
    return img

def apply_f4_salt_pepper(img, denom):
    x1, y1, x2, y2 = SEARCH_AREAS[denom][6]
    roi = img[y1:y2, x1:x2]
    noise = np.random.randint(0, 2, roi.shape[:2])
    roi[noise == 0] = [0, 0, 0]
    roi[noise == 1] = [255, 255, 255]
    img[y1:y2, x1:x2] = roi
    return img

def apply_f5_hsv_shift(img, denom):
    x1, y1, x2, y2 = SEARCH_AREAS[denom][5]
    roi = img[y1:y2, x1:x2]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:,:,0] = (hsv[:,:,0] + 90) % 180
    img[y1:y2, x1:x2] = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return img

def apply_f6_erode(img, denom):
    x1, y1, x2, y2 = SEARCH_AREAS[denom][4]
    roi = img[y1:y2, x1:x2]
    kernel = np.ones((5,5), np.uint8)
    img[y1:y2, x1:x2] = cv2.erode(roi, kernel, iterations=2)
    return img

def apply_f7_watermark_fill(img, denom):
    x1, y1, x2, y2 = SEARCH_AREAS[denom][7]
    cv2.rectangle(img, (x1, y1), (x2, y2), (200, 200, 200), -1)
    return img

def apply_f8_blind_dots(img, denom):
    x1, y1, x2, y2 = PROGRAMMATIC_COORDS[denom]['blind_dots']
    cv2.ellipse(img, ((x1+x2)//2, (y1+y2)//2), ((x2-x1)//2, (y2-y1)//4), 0, 0, 360, (150, 150, 150), -1)
    return img

def apply_f9_serial(img, denom):
    x1, y1, x2, y2 = PROGRAMMATIC_COORDS[denom]['asymmetric_serial']
    roi = img[y1:y2, x1:x2]
    resized = cv2.resize(roi, (roi.shape[1], roi.shape[0]//2))
    img[y1:y1+resized.shape[0], x1:x2] = resized
    img[y1+resized.shape[0]:y2, x1:x2] = (255,255,255)
    return img

def apply_f10_red_serial(img, denom):
    x1, y1, x2, y2 = PROGRAMMATIC_COORDS[denom]['vertical_red_serial']
    roi = img[y1:y2, x1:x2]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:,:,0] = (hsv[:,:,0] + 100) % 180
    img[y1:y2, x1:x2] = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return img

def apply_f11_security_thread(img, denom):
    x1, y1, x2, y2 = PROGRAMMATIC_COORDS[denom]['security_thread']
    cv2.line(img, ((x1+x2)//2, y1), ((x1+x2)//2, y2), (180, 180, 180), 8)
    return img

def apply_f12_tactile_lines(img, denom):
    x1, y1, x2, y2 = PROGRAMMATIC_COORDS[denom]['edge_lines']
    cv2.rectangle(img, (x1, y1), (x2, y1 + (y2-y1)//2), (255, 255, 255), -1)
    return img


def generate():
    base_dir = os.path.dirname(os.path.dirname(__file__))
    
    mutations = [
        ("F1", apply_f1_blur), ("F2", apply_f2_eq), ("F3", apply_f3_shear),
        ("F4", apply_f4_salt_pepper), ("F5", apply_f5_hsv_shift), ("F6", apply_f6_erode),
        ("F7", apply_f7_watermark_fill), ("F8", apply_f8_blind_dots), ("F9", apply_f9_serial),
        ("F10", apply_f10_red_serial), ("F11", apply_f11_security_thread), ("F12", apply_f12_tactile_lines),
    ]
    
    for denom in ['LKR_1000', 'LKR_5000']:
        gen_dir = os.path.join(base_dir, 'data', 'genuine', denom)
        out_dir = os.path.join(base_dir, 'data', 'synthetic_fakes', denom)
        
        if os.path.exists(out_dir):
            shutil.rmtree(out_dir)
        os.makedirs(out_dir, exist_ok=True)
        
        gen_paths = glob.glob(os.path.join(gen_dir, '*.jpg'))
        
        total = 0
        for path in gen_paths:
            name = os.path.splitext(os.path.basename(path))[0]
            base_img = cv2.imread(path)
            if base_img is None: continue
            
            for feat_name, mut_func in mutations:
                img_copy = base_img.copy()
                mutated = mut_func(img_copy, denom)
                out_path = os.path.join(out_dir, f"{name}_{feat_name}_fake.jpg")
                cv2.imwrite(out_path, mutated)
                total += 1
                
        print(f"Successfully generated {total} targeted synthetic fakes in {out_dir}")

if __name__ == "__main__":
    generate()
