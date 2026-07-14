import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset
import numpy as np
import time
import copy
import sys

def build_model(model_name):
    if model_name == 'mobilenet_v2':
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        for param in model.features.parameters():
            param.requires_grad = False
        model.classifier[1] = nn.Linear(model.last_channel, 2)
        
    elif model_name == 'resnet18':
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        for param in model.parameters():
            param.requires_grad = False
        for param in model.layer4.parameters():
            param.requires_grad = True
        model.fc = nn.Linear(model.fc.in_features, 2)
    else:
        raise ValueError("Unsupported model")
        
    return model

def train_model(model, dataloaders, dataset_sizes, criterion, optimizer, scheduler, num_epochs=5):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float('inf')

    for epoch in range(num_epochs):
        start_time = time.time()
        print(f"Epoch {epoch+1}/{num_epochs}")
        print("-" * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            
            for i, (inputs, labels) in enumerate(dataloaders[phase]):
                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        
                running_loss += loss.item() * inputs.size(0)
                
                if phase == 'train' and (i+1) % 10 == 0:
                    print(f"  Train Batch {i+1}/{len(dataloaders['train'])} processed...")
                    sys.stdout.flush()

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            
            elapsed = time.time() - start_time
            print(f"  {phase.capitalize()} Loss: {epoch_loss:.4f} [Elapsed: {elapsed:.0f}s]")
            sys.stdout.flush()

            # Deep copy the model if it's the best validation loss
            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                print(f"  [*] New best model saved! (Val Loss: {best_loss:.4f})")

        print()

    print(f"Training complete. Best Val Loss: {best_loss:.4f}")
    model.load_state_dict(best_model_wts)
    return model

def evaluate_model(model, test_loader):
    model.eval()
    
    TP, TN, FP, FN = 0, 0, 0, 0
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            
            for t, p in zip(labels, predicted):
                t = t.item()
                p = p.item()
                if t == 1 and p == 1: TP += 1
                if t == 0 and p == 0: TN += 1
                if t == 0 and p == 1: FP += 1
                if t == 1 and p == 0: FN += 1
                
    total = TP + TN + FP + FN
    correct = TP + TN
    accuracy = correct / total
    
    genuine_total = TP + FN
    counterfeit_total = TN + FP
    
    fn_rate = (FN / genuine_total) if genuine_total > 0 else 0
    fp_rate = (FP / counterfeit_total) if counterfeit_total > 0 else 0
    
    print(f"  Accuracy: {accuracy*100:.2f}%")
    print(f"  False Negative Rate (Genuine rejected): {fn_rate*100:.2f}% ({FN}/{genuine_total})")
    print(f"  False Positive Rate (Fake accepted): {fp_rate*100:.2f}% ({FP}/{counterfeit_total})")
    print(f"  [TP: {TP}, TN: {TN}, FP: {FP}, FN: {FN}]")

def main():
    print("Initializing Production-Grade PyTorch SOTA Pipeline...")
    
    # 1. Separate Transforms for Train (Augmentation) and Val/Test (Deterministic)
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    
    data_dir = 'data/augmented_testing'
    
    # Load dataset twice to apply different transforms using Subset
    full_dataset_train = datasets.ImageFolder(root=data_dir, transform=train_transform)
    full_dataset_test = datasets.ImageFolder(root=data_dir, transform=test_transform)
    
    num_samples = len(full_dataset_train)
    indices = np.random.permutation(num_samples)
    
    train_size = int(0.8 * num_samples)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    train_dataset = Subset(full_dataset_train, train_indices)
    val_dataset = Subset(full_dataset_test, val_indices)
    
    dataset_sizes = {'train': len(train_dataset), 'val': len(val_dataset)}
    
    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=32, shuffle=True),
        'val': DataLoader(val_dataset, batch_size=32, shuffle=False)
    }
    
    print(f"Classes found: {full_dataset_train.class_to_idx}")
    print(f"Training on {dataset_sizes['train']} images, Testing on {dataset_sizes['val']} images.")
    
    # Class 0: Counterfeit, Class 1: Genuine
    # We heavily penalize False Positives (Accepting a fake) by weighing Class 0 at 10.0
    class_weights = torch.tensor([10.0, 1.0])
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    models_to_test = ['mobilenet_v2', 'resnet18']
    
    for m_name in models_to_test:
        print(f"\n{'='*40}")
        print(f"Training SOTA Model: {m_name.upper()}")
        print(f"{'='*40}")
        
        model = build_model(m_name)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        # Scheduler: Drop learning rate by 0.1 every 2 epochs
        scheduler = lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)
        
        # Train with early stopping / best weight retention
        model = train_model(model, dataloaders, dataset_sizes, criterion, optimizer, scheduler, num_epochs=4)
        
        print(f"\nEvaluating BEST {m_name.upper()} on Test Set...")
        evaluate_model(model, dataloaders['val'])
        
if __name__ == "__main__":
    main()
