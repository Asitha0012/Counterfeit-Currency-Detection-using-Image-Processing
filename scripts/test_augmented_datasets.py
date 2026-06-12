import os
import glob
import sys

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(base_dir, 'src'))

from evaluate_lkr import analyze_lkr_note

def test_folder(folder_path, denom, expected_result):
    image_paths = glob.glob(os.path.join(folder_path, '*.jpg')) + glob.glob(os.path.join(folder_path, '*.png'))
    if not image_paths:
        print(f"No images found in {folder_path}")
        return
        
    print(f"\n--- Testing {len(image_paths)} images in {os.path.basename(folder_path)} ({denom}) ---")
    print(f"Expected Result: {'GENUINE (PASS)' if expected_result else 'COUNTERFEIT (FAIL)'}")
    
    correct_predictions = 0
    
    for img_path in image_paths:
        _, robust_verdict, message, score, _, _ = analyze_lkr_note(img_path, denom)
        if robust_verdict == expected_result:
            correct_predictions += 1
        else:
            print(f"  [FAILED] {os.path.basename(img_path)}: {message}")
            
    accuracy = (correct_predictions / len(image_paths)) * 100
    print(f"Result: {correct_predictions}/{len(image_paths)} behaved exactly as expected! (Accuracy: {accuracy:.1f}%)")

if __name__ == "__main__":
    augmented_base = os.path.join(base_dir, "data", "augmented_testing")
    
    print("\n" + "="*40 + "\nTESTING GENUINE NOTES (Should Pass)\n" + "="*40)
    test_folder(os.path.join(augmented_base, "Genuine", "LKR_1000"), "LKR_1000", expected_result=True)
    test_folder(os.path.join(augmented_base, "Genuine", "LKR_5000"), "LKR_5000", expected_result=True)
    
    print("\n" + "="*40 + "\nTESTING FAKE NOTES (Should Fail)\n" + "="*40)
    test_folder(os.path.join(augmented_base, "Fake", "LKR_1000"), "LKR_1000", expected_result=False)
    test_folder(os.path.join(augmented_base, "Fake", "LKR_5000"), "LKR_5000", expected_result=False)
