import cv2
import os

img = cv2.imread(r'data\genuine\LKR_500\500_G1.jpg')
h, w = img.shape[:2]
new_w = int(w * (500 / h))
img = cv2.resize(img, (new_w, 500))

out_dir = r'data\templates\LKR_500'
os.makedirs(out_dir, exist_ok=True)

# Using LKR_1000 percentages as the baseline
PCT = {
    1: (0.008, 0.710, 0.149, 0.988),
    2: (0.012, 0.534, 0.220, 0.988),
    3: (0.162, 0.638, 0.340, 0.948),
    4: (0.760, 0.200, 0.980, 0.750),
    5: (0.820, 0.020, 0.930, 0.250),
    6: (0.598, 0.788, 0.818, 0.982),
    7: (0.168, 0.318, 0.279, 0.758)
}

h, w = img.shape[:2]
for fid, (px1, py1, px2, py2) in PCT.items():
    x1, y1 = int(px1 * w), int(py1 * h)
    x2, y2 = int(px2 * w), int(py2 * h)
    
    # Optional: shrink the crop slightly to make it a better template for matching
    # (templates should be slightly smaller than search areas)
    margin_x = int((x2 - x1) * 0.1)
    margin_y = int((y2 - y1) * 0.1)
    
    crop = img[y1+margin_y : y2-margin_y, x1+margin_x : x2-margin_x]
    cv2.imwrite(os.path.join(out_dir, f'f{fid}_template.jpg'), crop)
    print(f"Saved f{fid}_template.jpg with shape {crop.shape}")

