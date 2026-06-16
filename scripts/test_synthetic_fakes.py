import os
import glob
import sys
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)
sys.path.append(os.path.join(base_dir, "src"))
from src.evaluate_lkr import analyze_lkr_note

def test_synthetic_fakes():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    print("========================================")
    print("TESTING TARGETED SYNTHETIC FAKES (Should Fail)")
    print("========================================")
    
    total_images = 0
    total_rejected = 0
    
    for denom in ['LKR_1000', 'LKR_5000']:
        fake_dir = os.path.join(base_dir, "data", "synthetic_fakes", denom)
        img_paths = glob.glob(os.path.join(fake_dir, '*.jpg'))
        
        if not img_paths:
            print(f"No synthetic fakes found for {denom}!")
            continue
            
        print(f"\n--- Testing {len(img_paths)} images in {denom} ---")
        
        rejected = 0
        for img in img_paths:
            features, is_genuine, msg, ssim_scores, ssim_avg, exec_time = analyze_lkr_note(img, denom)
            if not is_genuine:
                rejected += 1
            else:
                print(f"  [FAILED] {os.path.basename(img)}: GENUINE NOTE (Passed structural & programmatic verification) - THIS IS BAD, IT SHOULD BE FAKE")
                
        total_images += len(img_paths)
        total_rejected += rejected
        print(f"Result for {denom}: {rejected}/{len(img_paths)} rejected ({(rejected/len(img_paths))*100:.1f}%)")

    if total_images > 0:
        accuracy = (total_rejected / total_images) * 100
        print(f"\nOverall Result: {total_rejected}/{total_images} Synthetic Fakes were correctly identified and rejected as Counterfeits!")
        print(f"Overall Accuracy: {accuracy:.1f}%")

if __name__ == "__main__":
    test_synthetic_fakes()
