import cv2
import numpy as np
import os

def paint_out_thread(img):
    """Simulate a printed/photocopied thread by converting the thread region to grayscale.
    This destroys the color-shifting property (making it fail the system's test) while looking
    perfectly realistic to a human evaluator."""
    h, w = img.shape[:2]
    start_x = int(w * 0.49)
    end_x = int(w * 0.53)
    
    modified = img.copy()
    thread_region = modified[:, start_x:end_x]
    
    # Convert to grayscale to remove color-shifting properties
    gray_thread = cv2.cvtColor(thread_region, cv2.COLOR_BGR2GRAY)
    bgr_thread = cv2.cvtColor(gray_thread, cv2.COLOR_GRAY2BGR)
    
    # Darken slightly to look like printed ink
    bgr_thread = (bgr_thread * 0.85).astype(np.uint8)
    
    modified[:, start_x:end_x] = bgr_thread
    return modified

def shift_hue(img, shift_val=20):
    """Simulate printer color mismatch by shifting HSV hue."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    h = ((h.astype(np.int32) + shift_val) % 180).astype(np.uint8)
    hsv_shifted = cv2.merge((h, s, v))
    return cv2.cvtColor(hsv_shifted, cv2.COLOR_HSV2BGR)

def simulate_fake_dataset():
    base_dir = os.path.dirname(os.path.dirname(__file__))
    categories = [
        (os.path.join(base_dir, "data", "genuine", "500"), os.path.join(base_dir, "data", "synthetic", "500"), "500"),
        (os.path.join(base_dir, "data", "genuine", "2000"), os.path.join(base_dir, "data", "synthetic", "2000"), "2000")
    ]
    
    print("Starting generation of simulated counterfeit notes...")
    generated_count = 0
    
    for src_dir, dst_dir, denom in categories:
        if not os.path.exists(src_dir):
            continue
        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir)
            
        print(f"Synthesizing fakes from {src_dir} into {dst_dir}...")
        files = [f for f in os.listdir(src_dir) if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff"))]
        
        for file_name in files:
            file_path = os.path.join(src_dir, file_name)
            img = cv2.imread(file_path)
            if img is None:
                continue
                
            # Resize first to standardize coordinates
            if denom == "500":
                img = cv2.resize(img, (1167, 519))
            else:
                img = cv2.resize(img, (1165, 455))
                
            base_name = file_name.split(".")[0]
            
            # Type 1: Paint out thread (Thread tampering fake)
            img_thread = paint_out_thread(img)
            thread_path = os.path.join(dst_dir, f"sim_nothread_{base_name}.jpg")
            cv2.imwrite(thread_path, img_thread)
            
            # Type 2: Hue shifting (Color printer mismatch fake)
            img_hue = shift_hue(img, 20)
            hue_path = os.path.join(dst_dir, f"sim_color_{base_name}.jpg")
            cv2.imwrite(hue_path, img_hue)
            
            generated_count += 2
            
    print(f"Counterfeit synthesis complete. Generated {generated_count} simulated fakes successfully.")

if __name__ == '__main__':
    simulate_fake_dataset()
