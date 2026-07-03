===========================================================================
DATA AUGMENTATION PARAMETER SWEEP - CSCI 484
===========================================================================

PROJECT OVERVIEW
This project evaluates the impact of data augmentation on supervised image 
classification, specifically focusing on the trade-off between the amount 
of original labeled data and the intensity of augmentation [cite: 618-630].

1. DATA REORGANIZATION (data/ Directory) [cite: 1013-1023]
The original MVTec datasets (Carpet and Toothbrush) were reorganized from 
their anomaly-detection structure into a supervised binary format [cite: 680-691, 961-973]:
- Class 0: All "good" images from the original train and test sets[cite: 688].
- Class 1: All defective images (e.g., hole, color, scratch)[cite: 689].
- The Cat vs Dog dataset was similarly flattened into Class 0 (Cat) and 
  Class 1 (Dog) [cite: 652-667].
- All original ground_truth folders were ignored as they are not used for 
  supervised classification[cite: 646, 685].

2. INSTALLATION & DEPENDENCIES [cite: 1041]
The following Python libraries are required to run the experimental scripts:
- torch & torchvision (Model architecture and augmentations) [cite: 177-184]
- pandas (Result logging and CSV management)
- scikit-learn (70/30 data splitting)
- seaborn & matplotlib (Heatmap and graph generation)

3. HOW TO RUN THE CODE (code/ Directory) [cite: 1040-1042]

Step 1: Flatten the Datasets
Run the following to reorganize your raw folders into the binary structure:
> python prepare_all_data.py

Step 2: Run the Experimental Sweep
Run the trainer for each dataset by updating the DATASET_PATH in the script:
> python sweep_trainer.py
- This performs a fixed 70/30 split before any augmentation [cite: 698-706].
- It evaluates 8 training fractions (5% to 100%) and 5 repeat levels (0 to 8) [cite: 814-853].

Step 3: Generate Visualizations
After the CSV results appear, run the analysis script to create heatmaps, 
confusion matrices, and comparison graphs [cite: 912-917]:
> python generate_analysis.py

4. EXPERIMENTAL DESIGN NOTES [cite: 1067-1070]
- Model: ResNet-18 trained from scratch (weights=None)[cite: 865, 871].
- Baseline Augmentation: Random crop, horizontal flip, and rotation [cite: 763-768].
- Fairness: The 30% test set remains identical for all experiments [cite: 858-859, 975-979].
- Metrics: We record accuracy, training time, and confusion matrix data [cite: 894-909].

===========================================================================