import os
import shutil
from pathlib import Path

def setup_dir(path):
    """Creates a directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)

def flatten_mvtec(dataset_name, source_root, output_root):
    """
    Reorganizes MVTec anomaly data (Carpet/Toothbrush) into binary folders.
    Class 0: Good images | Class 1: All defective images
    """
    class_0_dir = os.path.join(output_root, dataset_name, 'class_0')
    class_1_dir = os.path.join(output_root, dataset_name, 'class_1')
    setup_dir(class_0_dir)
    setup_dir(class_1_dir)
    
    dataset_path = os.path.join(source_root, dataset_name)
    if not os.path.exists(dataset_path):
        print(f"Skipping {dataset_name}: Source path not found.")
        return

    # 1. Collect Good Images (Class 0)
    # Checks both train and test 'good' folders
    for folder in ['train', 'test']:
        src = os.path.join(dataset_path, folder, 'good')
        if os.path.exists(src):
            for img in os.listdir(src):
                if img.lower().endswith(('.png', '.jpg', '.jpeg')):
                    shutil.copy2(os.path.join(src, img), 
                                 os.path.join(class_0_dir, f"{folder}_{img}"))

    # 2. Collect Defective Images (Class 1)
    # Gathers all folders in 'test' EXCEPT 'good'
    test_path = os.path.join(dataset_path, 'test')
    if os.path.exists(test_path):
        defects = [d for d in os.listdir(test_path) if d != 'good' and os.path.isdir(os.path.join(test_path, d))]
        for defect_type in defects:
            src = os.path.join(test_path, defect_type)
            for img in os.listdir(src):
                if img.lower().endswith(('.png', '.jpg', '.jpeg')):
                    shutil.copy2(os.path.join(src, img), 
                                 os.path.join(class_1_dir, f"{defect_type}_{img}"))

    print(f"Finished {dataset_name}: {len(os.listdir(class_0_dir))} Good, {len(os.listdir(class_1_dir))} Bad")

def flatten_cat_dog(source_dir, output_root):
    """
    Reorganizes Cat vs Dog into a flat binary structure.
    Class 0: Cats | Class 1: Dogs
    """
    class_0_dir = os.path.join(output_root, 'cat_dog', 'class_0')
    class_1_dir = os.path.join(output_root, 'cat_dog', 'class_1')
    setup_dir(class_0_dir)
    setup_dir(class_1_dir)

    if not os.path.exists(source_dir):
        print("Skipping Cat/Dog: Source path not found.")
        return

    for root, _, files in os.walk(source_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                # Assignment requires NO snakes - only Cat/Dog [cite: 645]
                if 'cat' in root.lower() or 'cat' in file.lower():
                    shutil.copy2(os.path.join(root, file), os.path.join(class_0_dir, file))
                elif 'dog' in root.lower() or 'dog' in file.lower():
                    shutil.copy2(os.path.join(root, file), os.path.join(class_1_dir, file))

    print(f"Finished Cat/Dog: {len(os.listdir(class_0_dir))} Cats, {len(os.listdir(class_1_dir))} Dogs")

if __name__ == "__main__":
    # --- CONFIGURATION: Update these paths to match your folders ---
    # Based on your screenshots:
    MVTEC_SOURCE = "./"            # Path containing 'carpet' and 'toothbrush'
    CATDOG_SOURCE = "./animal_split_data" 
    OUTPUT_FOLDER = "./reorganized_binary_data"
    # -------------------------------------------------------------

    setup_dir(OUTPUT_FOLDER)
    
    # Process all three required datasets [cite: 641, 642, 643]
    flatten_mvtec("carpet", MVTEC_SOURCE, OUTPUT_FOLDER)
    flatten_mvtec("toothbrush", MVTEC_SOURCE, OUTPUT_FOLDER)
    flatten_cat_dog(CATDOG_SOURCE, OUTPUT_FOLDER)
    
    print(f"\nAll data reorganized in: {os.path.abspath(OUTPUT_FOLDER)}")