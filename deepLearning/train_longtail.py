import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import timm
from tqdm import tqdm
from image_preprocess_for_longtail import prepare_longtail_datasets
import os

from utils.imbalanced_utils import get_class_weighted_loss, get_weighted_sampler

# Configuration
DATASET_PATH = 'dataset/aerial_landscapes'
MODEL_NAME = 'seresnet50'
BATCH_SIZE = 32
NUM_EPOCHS = 30
NUM_CLASSES = 15
LEARNING_RATE = 1e-4
SEED = 18
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRAIN_RATIO = 0.8
CHECKPOINT_DIR = "checkpoints_longtails"
USE_CLASS_WEIGHT = False   # Use class weight
USE_RESAMPLING = False    # Use weighted sampler
DATASET_AUG = False     # dataset augmentation
if DATASET_AUG:
    if USE_CLASS_WEIGHT and USE_RESAMPLING:
        file_name_extension = 'both'
    elif USE_CLASS_WEIGHT:
        file_name_extension = 'reweighting'
    elif USE_RESAMPLING:
        file_name_extension = 'resampling'
    else:
        file_name_extension = 'only_aug_data'
else:
    file_name_extension = 'without_any_aug'

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Set random seed
torch.manual_seed(SEED)

# Load dataset
train_dataset, test_dataset, class_counts = prepare_longtail_datasets(
    dataset_path=DATASET_PATH,
    train_ratio=TRAIN_RATIO,
    seed=SEED,
    dataset_aug=DATASET_AUG,
)

if USE_RESAMPLING:
    labels = [train_dataset[i][1] for i in range(len(train_dataset))]
    sampler = get_weighted_sampler(labels)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler)
else:
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"Dataset loaded successfully: Train={len(train_dataset)}, Test={len(test_dataset)}")

# Build SENet model
model = timm.create_model(MODEL_NAME, pretrained=True)
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
model = model.to(DEVICE)

# Loss function and optimizer
if USE_CLASS_WEIGHT:
    criterion = get_class_weighted_loss(class_counts, DEVICE)
    print("Using class weight")
else:
    criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Training + evaluation functions
def train_one_epoch(model, dataloader, optimizer, criterion, device, epoch):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for inputs, labels in tqdm(dataloader, desc=f"[Epoch {epoch+1}]"):
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    train_acc = correct / total
    print(f"[Epoch {epoch+1}] Train Loss: {running_loss / len(dataloader):.4f}, Acc: {train_acc*100:.2f}%")
    return train_acc

def evaluate(model, dataloader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Evaluating"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    return correct / total

best_acc = 0.0  # Track best accuracy

# Start training
train_accs, val_accs = [], []
for epoch in range(NUM_EPOCHS):
    train_acc = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE, epoch)
    val_acc = evaluate(model, test_loader, DEVICE)
    print(f"[Epoch {epoch+1}] Test Accuracy: {val_acc*100:.2f}%")

    train_accs.append(train_acc)
    val_accs.append(val_acc)
    # Save current epoch model
    checkpoint_path = os.path.join(CHECKPOINT_DIR, f"epoch_{epoch+1}_acc_{val_acc:.2f}.pth")
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Model saved to {checkpoint_path}")

    # Save best model
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), "best_model_longtail_"+file_name_extension+".pth")
        print(f"Epoch {epoch+1}: New best model saved ({best_acc*100:.2f}%)")

train_accs_filename = "train_accs_longtail_"+file_name_extension+".npy"
val_accs_filename = "val_accs_longtail_"+file_name_extension+".npy"
np.save(train_accs_filename, np.array(train_accs))
np.save(val_accs_filename, np.array(val_accs))

print("Train/Val accuracy arrays saved as '"+train_accs_filename+"' and '"+val_accs_filename+"'")