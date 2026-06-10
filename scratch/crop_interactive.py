import cv2
import os

def interactive_crop_multiple(denom, feature_num, feature_name):
    print(f"\n=======================================")
    print(f" CROPPING {denom} - Feature {feature_num}: {feature_name}")
    print(f"=======================================")
    
    out_dir = f'data/templates/{denom}/Feature {feature_num}'
    os.makedirs(out_dir, exist_ok=True)
    
    prefix = '1322' if denom == 'LKR_1000' else '1326'
    
    for i in range(1, 11):
        img_name = f'Scanned_20260608-{prefix}-{i:02d}.jpg'
        img_path = f'data/genuine/{denom}/{img_name}'
        
        img = cv2.imread(img_path)
        if img is None:
            continue
            
        display_scale = 1.0
        if img.shape[0] > 800:
            display_scale = 800.0 / img.shape[0]
            display_img = cv2.resize(img, (int(img.shape[1] * display_scale), int(img.shape[0] * display_scale)))
        else:
            display_img = img.copy()

        window_name = f"Draw box around {feature_name} - {denom} Image {i}/10"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        roi = cv2.selectROI(window_name, display_img, showCrosshair=True, fromCenter=False)
        cv2.destroyAllWindows()
        
        if roi[2] > 0 and roi[3] > 0:
            x = int(roi[0] / display_scale)
            y = int(roi[1] / display_scale)
            w = int(roi[2] / display_scale)
            h = int(roi[3] / display_scale)
            
            cropped_feature = img[y:y+h, x:x+w]
            save_path = os.path.join(out_dir, f'{i}.jpg')
            cv2.imwrite(save_path, cropped_feature)
            print(f"✅ Saved Image {i}/10 -> {save_path}")
        else:
            print(f"❌ Skipped Image {i}/10")

if __name__ == "__main__":
    print("INSTRUCTIONS: Draw a box and press ENTER. Press 'c' to cancel and redraw.")
    
    # We already did Feature 1 (Blind Dots).
    # Now we need Features 2, 3, 4, 5 for both notes.
    
    features = {
        1: "Lion Emblem (Top Right)",
        2: "Bird (Right side)",
        3: "Butterfly (Bottom Left)",
        4: "1000 or 5000 Numeral (Top Left or Bottom Right)",
        5: "Micro-Printing Area (Tiny CBSL text)"
    }
    
    for f_num, f_name in features.items():
        interactive_crop_multiple('LKR_1000', f_num, f_name)
        
    for f_num, f_name in features.items():
        interactive_crop_multiple('LKR_5000', f_num, f_name)
        
    print("\n🎉 ALL DONE! You have successfully cropped all necessary templates!")
