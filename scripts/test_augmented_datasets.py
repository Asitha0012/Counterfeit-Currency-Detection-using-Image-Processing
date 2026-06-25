import os
import glob
import sys

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(base_dir, 'src'))

from evaluate_lkr import analyze_lkr_note
from evaluate_matrices import calculate_metrics, print_confusion_matrix

def test_folder(folder_path, denom, expected_result):
    image_paths = glob.glob(os.path.join(folder_path, '*.jpg')) + glob.glob(os.path.join(folder_path, '*.png'))
    if not image_paths:
        return [], []
        
    print(f"\n--- Testing {len(image_paths)} images in {os.path.basename(folder_path)} ({denom}) ---")
    
    y_true = []
    y_pred = []

    for img_path in image_paths:
        flat_verdict, robust_verdict, message, score, feature_statuses, feature_images = analyze_lkr_note(img_path, denom)
        
        true_label = "Genuine" if expected_result else "Fake"
        pred_label = "Genuine" if robust_verdict else "Fake"
        
        y_true.append(true_label)
        y_pred.append(pred_label)
        
        if robust_verdict != expected_result:
            print(f"  [FAILED] {os.path.basename(img_path)}: {message}")
            
    accuracy = sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(image_paths) * 100
    print(f"Result: {accuracy:.1f}% behaved exactly as expected.")
            
    return y_true, y_pred

if __name__ == "__main__":
    augmented_base = os.path.join(base_dir, "data", "augmented_testing")
    
    y_true_all = []
    y_pred_all = []
    
    print("\n" + "="*40 + "\nTESTING GENUINE NOTES (Should Pass)\n" + "="*40)
    yt, yp = test_folder(os.path.join(augmented_base, "Genuine", "LKR_1000"), "LKR_1000", expected_result=True)
    y_true_all.extend(yt); y_pred_all.extend(yp)
    
    yt, yp = test_folder(os.path.join(augmented_base, "Genuine", "LKR_5000"), "LKR_5000", expected_result=True)
    y_true_all.extend(yt); y_pred_all.extend(yp)
    
    print("\n" + "="*40 + "\nTESTING FAKE NOTES (Should Fail)\n" + "="*40)
    yt, yp = test_folder(os.path.join(augmented_base, "Fake", "LKR_1000"), "LKR_1000", expected_result=False)
    y_true_all.extend(yt); y_pred_all.extend(yp)
    
    yt, yp = test_folder(os.path.join(augmented_base, "Fake", "LKR_5000"), "LKR_5000", expected_result=False)
    y_true_all.extend(yt); y_pred_all.extend(yp)
    
    TP, TN, FP, FN, precision, recall, f1, accuracy = calculate_metrics(y_true_all, y_pred_all)
    print_confusion_matrix("Augmented Dataset Full Results", TP, TN, FP, FN, precision, recall, f1, accuracy)
