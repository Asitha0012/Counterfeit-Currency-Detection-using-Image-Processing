import os
import glob
import sys
import concurrent.futures

# We must add 'src' to path to make evaluate_lkr imports work
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

from evaluate_lkr import analyze_lkr_note

import os
import glob
import sys
import concurrent.futures

# We must add 'src' to path to make evaluate_lkr imports work
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

from evaluate_lkr import analyze_lkr_note

def process_single_image(args):
    img_path, denom, ds_type = args
    
    filename = os.path.basename(img_path)
    
    # Extract category based on the dataset type prefix
    parts = filename.split(f"_{ds_type}_")
    if len(parts) > 1:
        category = parts[1].split('_')[0]
    else:
        category = 'Unknown'
        
    flat_verdict, robust_verdict, message, score, _, _ = analyze_lkr_note(img_path, denom)
    
    # Passing logic depends on dataset type
    if ds_type == 'genuine':
        passed = robust_verdict == True
    else:
        passed = robust_verdict == False  # Counterfeit pass if the system fails them
        
    return denom, ds_type, category, passed, score

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    aug_base = os.path.join(base_dir, "data", "augmented_testing")
    
    denoms = ['LKR_500', 'LKR_1000', 'LKR_5000']
    dataset_types = ['genuine', 'counterfeit']
    
    args_list = []
    
    for ds_type in dataset_types:
        ds_path = os.path.join(aug_base, ds_type.capitalize())
        for denom in denoms:
            path = os.path.join(ds_path, denom)
            files = glob.glob(os.path.join(path, "*.jpg")) + glob.glob(os.path.join(path, "*.png"))
            for f in files:
                args_list.append((f, denom, ds_type))
                
    print(f"Starting parallel evaluation of {len(args_list)} augmented notes using ProcessPoolExecutor...")
    
    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=6) as executor:
        results = list(executor.map(process_single_image, args_list))
        
    print("Evaluation complete. Aggregating statistics...")
    
    # Initialize statistics trackers
    stats = {
        'genuine': {'total': 0, 'passed': 0, 'denoms': {d: {'total': 0, 'passed': 0} for d in denoms}, 'cats': {}},
        'counterfeit': {'total': 0, 'passed': 0, 'denoms': {d: {'total': 0, 'passed': 0} for d in denoms}, 'cats': {}}
    }
    
    for denom, ds_type, cat, passed, score in results:
        ds_stats = stats[ds_type]
        ds_stats['total'] += 1
        ds_stats['denoms'][denom]['total'] += 1
        
        if cat not in ds_stats['cats']:
            ds_stats['cats'][cat] = {'total': 0, 'passed': 0}
        ds_stats['cats'][cat]['total'] += 1
        
        if passed:
            ds_stats['passed'] += 1
            ds_stats['denoms'][denom]['passed'] += 1
            ds_stats['cats'][cat]['passed'] += 1
            
    # Write to a report file
    report_path = os.path.join(base_dir, "pass_rates_report.md")
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Comprehensive Augmented Dataset Pass Rates Report\n\n")
        f.write("> **Note**: For Genuine notes, a 'Pass' means the system identified it as TRUE. For Counterfeit notes, a 'Pass' means the system successfully rejected it as FALSE.\n\n")
        
        for ds_type in dataset_types:
            ds_stats = stats[ds_type]
            total = ds_stats['total']
            passed = ds_stats['passed']
            pct = (passed / total * 100) if total > 0 else 0
            
            f.write(f"## {ds_type.capitalize()} Notes Analysis\n\n")
            f.write(f"**Total Notes Evaluated**: {total}\n")
            f.write(f"**Overall {ds_type.capitalize()} Pass Rate**: {passed}/{total} ({pct:.2f}%)\n\n")
            
            f.write(f"### Pass Rate by Denomination\n\n")
            f.write("| Denomination | Passed | Total | Pass Percentage |\n")
            f.write("| :--- | :--- | :--- | :--- |\n")
            for denom in denoms:
                d_total = ds_stats['denoms'][denom]['total']
                d_passed = ds_stats['denoms'][denom]['passed']
                d_pct = (d_passed / d_total * 100) if d_total > 0 else 0
                f.write(f"| {denom} | {d_passed} | {d_total} | {d_pct:.2f}% |\n")
                
            f.write("\n### Pass Rate by Augmentation Category\n\n")
            f.write("| Category | Passed | Total | Pass Percentage |\n")
            f.write("| :--- | :--- | :--- | :--- |\n")
            for cat in sorted(ds_stats['cats'].keys()):
                c_total = ds_stats['cats'][cat]['total']
                c_passed = ds_stats['cats'][cat]['passed']
                c_pct = (c_passed / c_total * 100) if c_total > 0 else 0
                f.write(f"| {cat} | {c_passed} | {c_total} | {c_pct:.2f}% |\n")
            f.write("\n---\n\n")
            
    print("\n" + "="*50)
    print("EVALUATION RESULTS SUMMARY:")
    print("="*50)
    
    for ds_type in dataset_types:
        ds_stats = stats[ds_type]
        total = ds_stats['total']
        passed = ds_stats['passed']
        pct = (passed / total * 100) if total > 0 else 0
        
        print(f"\n--- {ds_type.upper()} DATASET ---")
        
        print(f"{'Denomination':<15}{'Passed':<10}{'Total':<10}{'Pass Percentage'}")
        for denom in denoms:
            d_total = ds_stats['denoms'][denom]['total']
            d_passed = ds_stats['denoms'][denom]['passed']
            d_pct = (d_passed / d_total * 100) if d_total > 0 else 0
            print(f"{denom:<15}{d_passed:<10}{d_total:<10}{d_pct:.2f}%")
        print(f"{'OVERALL':<15}{passed:<10}{total:<10}{pct:.2f}%")
            
        print(f"\n{'Category':<25}{'Passed':<10}{'Total':<10}{'Pass Percentage'}")
        for cat in sorted(ds_stats['cats'].keys()):
            c_total = ds_stats['cats'][cat]['total']
            c_passed = ds_stats['cats'][cat]['passed']
            c_pct = (c_passed / c_total * 100) if c_total > 0 else 0
            print(f"{cat:<25}{c_passed:<10}{c_total:<10}{c_pct:.2f}%")
        print(f"{'OVERALL':<25}{passed:<10}{total:<10}{pct:.2f}%")
        print("\n" + "-" * 50)
        
    print(f"\nDetailed comprehensive report saved to {report_path}")

if __name__ == "__main__":
    main()
