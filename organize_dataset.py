import os
import shutil
import glob

def organize_dataset():
    base_dir = os.path.join('data', 'augmented_testing')
    
    for category in ['Genuine', 'Fake']:
        cat_dir = os.path.join(base_dir, category)
        
        # Create subdirectories
        for denom in ['LKR_500', 'LKR_1000', 'LKR_5000']:
            os.makedirs(os.path.join(cat_dir, denom), exist_ok=True)
            
        # Move files
        for img_path in glob.glob(os.path.join(cat_dir, '*.*')):
            if os.path.isdir(img_path):
                continue
                
            filename = os.path.basename(img_path)
            
            if filename.startswith('500_'):
                target_dir = os.path.join(cat_dir, 'LKR_500')
            elif filename.startswith('1000_'):
                target_dir = os.path.join(cat_dir, 'LKR_1000')
            elif filename.startswith('5000_'):
                target_dir = os.path.join(cat_dir, 'LKR_5000')
            else:
                continue
                
            shutil.move(img_path, os.path.join(target_dir, filename))
            
    print("Files successfully organized into denomination folders!")

if __name__ == '__main__':
    organize_dataset()
