import cv2
import numpy as np
import os
import random

def rotate_image(image, angle):
    """Rotate image by a given angle in degrees."""
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return rotated

def scale_image(image, scale_factor):
    """Scale/zoom image by a given factor."""
    h, w = image.shape[:2]
    new_h, new_w = int(h * scale_factor), int(w * scale_factor)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    # Crop or pad to keep the original size
    if scale_factor > 1.0:
        dy = (new_h - h) // 2
        dx = (new_w - w) // 2
        cropped = resized[dy:dy+h, dx:dx+w]
        # In case of rounding errors
        if cropped.shape[:2] != (h, w):
            cropped = cv2.resize(cropped, (w, h))
        return cropped
    else:
        padded = np.zeros_like(image)
        dy = (h - new_h) // 2
        dx = (w - new_w) // 2
        padded[dy:dy+new_h, dx:dx+new_w] = resized
        return padded

def shift_brightness(image, value):
    """Shift brightness of the BGR/gray image."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    
    # Adjust value channel with clipping
    v = np.clip(v.astype(np.int32) + value, 0, 255).astype(np.uint8)
    
    final_hsv = cv2.merge((h, s, v))
    return cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)

def add_gaussian_noise(image, mean=0, sigma=15):
    """Add Gaussian noise to the image."""
    noise = np.random.normal(mean, sigma, image.shape).astype(np.int32)
    noisy_image = np.clip(image.astype(np.int32) + noise, 0, 255).astype(np.uint8)
    return noisy_image
def augment_template_datasets():
    base_dir_parent = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "templates")
    base_dirs = [
        os.path.join(base_dir_parent, "500"),
        os.path.join(base_dir_parent, "2000")
    ]
    
    print("Starting active data augmentation process...")
    augmented_count = 0
    
    for base_dir in base_dirs:
        if not os.path.exists(base_dir):
            print(f"Directory not found: {base_dir}")
            continue
            
        print(f"Processing directory: {base_dir}")
        for feature_folder in sorted(os.listdir(base_dir)):
            feature_path = os.path.join(base_dir, feature_folder)
            if not os.path.isdir(feature_path):
                continue
                
            # Find original templates (1.jpg to 6.jpg or matching files)
            files = [f for f in os.listdir(feature_path) if f.endswith(".jpg") and not f.startswith("aug_")]
            
            for file_name in files:
                file_path = os.path.join(feature_path, file_name)
                img = cv2.imread(file_path)
                if img is None:
                    continue
                
                # Apply 1: Rotation
                angle = random.choice([-8, -5, 5, 8])
                img_rot = rotate_image(img, angle)
                rot_name = f"aug_rot_{file_name}"
                cv2.imwrite(os.path.join(feature_path, rot_name), img_rot)
                
                # Apply 2: Scaling
                scale = random.choice([0.92, 1.08])
                img_scale = scale_image(img, scale)
                scale_name = f"aug_scale_{file_name}"
                cv2.imwrite(os.path.join(feature_path, scale_name), img_scale)
                
                # Apply 3: Brightness shift
                shift = random.choice([-25, 25])
                img_bright = shift_brightness(img, shift)
                bright_name = f"aug_bright_{file_name}"
                cv2.imwrite(os.path.join(feature_path, bright_name), img_bright)
                
                # Apply 4: Gaussian Noise
                img_noise = add_gaussian_noise(img)
                noise_name = f"aug_noise_{file_name}"
                cv2.imwrite(os.path.join(feature_path, noise_name), img_noise)
                
                augmented_count += 4
                
    print(f"Data augmentation complete. Created {augmented_count} augmented templates successfully.")

if __name__ == "__main__":
    augment_template_datasets()
