import cv2
import os
import sys
sys.path.append('src')
from align_note import align_note

def auto_crop_feature(denom, feature_num, feature_name):
    print(f"\n=======================================")
    print(f" AUTO-CROPPING {denom} - Feature {feature_num}: {feature_name}")
    print(f"=======================================")
    
    out_dir = f'data/templates/{denom}/Feature {feature_num}'
    os.makedirs(out_dir, exist_ok=True)
    
    prefix = '1322' if denom == 'LKR_1000' else '1326'
    
    # 1. Load the FIRST image to get the coordinate
    img1_name = f'Scanned_20260608-{prefix}-01.jpg'
    img1_path = f'data/genuine/{denom}/{img1_name}'
    img1_raw = cv2.imread(img1_path)
    
    if img1_raw is None:
        print(f"Error loading {img1_path}")
        return

    # 🔴 CRITICAL FIX: Align the note FIRST so it perfectly matches the standard dimensions!
    img1 = align_note(img1_raw, denom)

    display_scale = 1.0
    if img1.shape[0] > 800:
        display_scale = 800.0 / img1.shape[0]
        display_img = cv2.resize(img1, (int(img1.shape[1] * display_scale), int(img1.shape[0] * display_scale)))
    else:
        display_img = img1.copy()

    print(f"INSTRUCTIONS: Draw a box around {feature_name} on this FIRST image and press ENTER.")
    window_name = f"Draw box around {feature_name} - {denom}"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    roi = cv2.selectROI(window_name, display_img, showCrosshair=True, fromCenter=False)
    cv2.destroyAllWindows()
    
    if roi[2] > 0 and roi[3] > 0:
        # Get actual coordinates
        x = int(roi[0] / display_scale)
        y = int(roi[1] / display_scale)
        w = int(roi[2] / display_scale)
        h = int(roi[3] / display_scale)
        
        print(f"✅ Box selected at ({x}, {y}, {x+w}, {y+h}). Now automatically cropping all 10 notes...")
        
        # 2. Loop through all 10 images and crop that exact same box!
        for i in range(1, 11):
            img_name = f'Scanned_20260608-{prefix}-{i:02d}.jpg'
            img_path = f'data/genuine/{denom}/{img_name}'
            
            img_raw = cv2.imread(img_path)
            if img_raw is None:
                continue
                
            # 🔴 CRITICAL FIX: Align each note BEFORE cropping!
            img = align_note(img_raw, denom)
            
            cropped_feature = img[y:y+h, x:x+w]
            save_path = os.path.join(out_dir, f'{i}.jpg')
            cv2.imwrite(save_path, cropped_feature)
            print(f"  -> Saved {save_path}")
            
    else:
        print("❌ Canceled.")

if __name__ == "__main__":
    auto_crop_feature('LKR_1000', 3, 'Note Value Numeral')
    auto_crop_feature('LKR_5000', 3, 'Note Value Numeral')
