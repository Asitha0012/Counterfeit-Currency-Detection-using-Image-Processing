import cv2

def get_coordinates(denom, img_path, feature_name):
    print(f"\n--- Finding {feature_name} for {denom} ---")
    img = cv2.imread(img_path)
    if img is None:
        print(f"Error loading {img_path}")
        return
        
    display_scale = 1.0
    if img.shape[0] > 800:
        display_scale = 800.0 / img.shape[0]
        display_img = cv2.resize(img, (int(img.shape[1] * display_scale), int(img.shape[0] * display_scale)))
    else:
        display_img = img.copy()

    print(f"Draw a tight box around the ENTIRE {feature_name} and press ENTER.")
    window_name = f"Select {feature_name} - {denom}"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    roi = cv2.selectROI(window_name, display_img, showCrosshair=True, fromCenter=False)
    cv2.destroyAllWindows()
    
    if roi[2] > 0 and roi[3] > 0:
        x = int(roi[0] / display_scale)
        y = int(roi[1] / display_scale)
        w = int(roi[2] / display_scale)
        h = int(roi[3] / display_scale)
        print(f"\n✅ {denom} {feature_name} Coordinates: ({x}, {y}, {x+w}, {y+h})")
    else:
        print("❌ Canceled.")

if __name__ == "__main__":
    # LKR 1000
    get_coordinates('LKR_1000', 'data/genuine/LKR_1000/Scanned_20260608-1322-01.jpg', 'Security Thread')
    # LKR 5000
    get_coordinates('LKR_5000', 'data/genuine/LKR_5000/Scanned_20260608-1326-01.jpg', 'Security Thread')
