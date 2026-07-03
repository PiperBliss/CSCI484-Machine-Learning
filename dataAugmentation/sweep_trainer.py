import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, ConcatDataset
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import pandas as pd
import os
import time

# --- CONFIGURATION [cite: 641-643] ---
# Change this for each run: 'carpet', 'toothbrush', or 'cat_dog'
DATASET_PATH = "./reorganized_binary_data/cat_dog" 
TRAIN_FRACTIONS = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0] # [cite: 814-820]
REPEAT_LEVELS = [0, 1, 2, 4, 8] # [cite: 844-848]
BATCH_SIZE = 32
EPOCHS = 10 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- TRANSFORMS [cite: 52-53, 763-768] ---
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
])

def train_and_evaluate(train_loader, test_loader):
    # Initialize ResNet-18 [cite: 865]
    model = torchvision.models.resnet18(weights=None) 
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2) # Binary classification [cite: 644]
    model = model.to(DEVICE)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    # Record Training Time [cite: 906]
    start_time = time.time()
    for epoch in range(EPOCHS):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    training_duration = time.time() - start_time
            
    # Evaluation on FIXED test set [cite: 704-706, 807-808]
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    # Collect Confusion Matrix data [cite: 908]
    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    accuracy = (100 * (tp + tn) / len(all_labels))
    
    return accuracy, training_duration, tn, fp, fn, tp

def run_sweep():
    if not os.path.exists(DATASET_PATH):
        print(f"Error: {DATASET_PATH} not found.")
        return

    full_dataset = ImageFolder(root=DATASET_PATH)
    indices = list(range(len(full_dataset)))
    
    # Mandatory 70/30 split BEFORE augmentation [cite: 698-702, 882-884]
    train_idx, test_idx = train_test_split(
        indices, test_size=0.30, random_state=42, stratify=full_dataset.targets
    )
    
    test_ds = Subset(ImageFolder(root=DATASET_PATH, transform=test_transform), test_idx)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
    
    all_results = []
    dataset_name = os.path.basename(DATASET_PATH)

    for frac in TRAIN_FRACTIONS:
        # Calculate actual N based on the 70% partition [cite: 724-728, 838-841]
        subset_size = int(len(train_idx) * frac)
        current_train_indices = train_idx[:subset_size]
        
        for r in REPEAT_LEVELS:
            print(f"--- {dataset_name} | Frac: {frac*100}% | Repeat: {r} ---")
            
            base_train = Subset(ImageFolder(root=DATASET_PATH, transform=train_transform), current_train_indices)
            
            # Apply Augmentation Repeats [cite: 781-783, 842-848]
            train_copies = [base_train]
            for _ in range(r):
                train_copies.append(base_train)
            
            final_train_ds = ConcatDataset(train_copies)
            train_loader = DataLoader(final_train_ds, batch_size=BATCH_SIZE, shuffle=True)
            
            # Run Training and Record all Required Metrics [cite: 894-904]
            acc, duration, tn, fp, fn, tp = train_and_evaluate(train_loader, test_loader)
            
            all_results.append({
                'dataset_name': dataset_name,
                'class_names': str(full_dataset.classes),
                'total_original_image_count': len(full_dataset),
                'original_training_partition_count': len(train_idx),
                'original_testing_partition_count': len(test_idx),
                'training_fraction_used': frac,
                'actual_number_original_training_images': subset_size,
                'augmentation_repeat_level': r,
                'effective_training_size': len(final_train_ds),
                'test_accuracy': acc,
                'training_time_sec': duration,
                'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp # Data for Confusion Matrix [cite: 908]
            })
            print(f"Done -> Acc: {acc:.2f}% | Time: {duration:.2f}s")
            
    # Save CSV for analysis and trade-off discussion [cite: 923-929]
    df = pd.DataFrame(all_results)
    csv_name = f"results_{dataset_name}.csv"
    df.to_csv(csv_name, index=False)
    print(f"\nFinal results saved to {csv_name}")

if __name__ == "__main__":
    run_sweep()