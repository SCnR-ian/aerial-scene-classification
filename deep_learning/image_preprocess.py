from collections import defaultdict
import numpy as np
from torchvision import transforms, datasets
from torch.utils.data import Subset, Dataset
import torch
import random


# Custom wrapper Dataset
class MyDataset(Dataset):
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        image, label = self.subset[index]
        image = self.transform(image)
        return image, label

    def __len__(self):
        return len(self.subset)

def get_train_transform():
    return transforms.Compose([
        transforms.Resize((224,224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.3,  
                                contrast=0.3,
                                saturation=0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])                              
    ])

def get_test_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
    ])


# Load data from ImageFolder, split into train/test by class, apply respective transforms
def prepare_datasets(dataset_path, train_ratio=0.8, seed=42):
        
    raw_dataset = datasets.ImageFolder(dataset_path)
    class_indices = defaultdict(list)
    torch.manual_seed(seed)

    for idx, (_, label) in enumerate(raw_dataset):
        class_indices[label].append(idx)

    train_indices, test_indices = [], []
    for label, indices in class_indices.items():
        indices = torch.tensor(indices)[torch.randperm(len(indices))]
        n_train = int(train_ratio * len(indices))
        train_indices += indices[:n_train].tolist()
        test_indices += indices[n_train:].tolist()

    # Build raw Subsets (no transform)
    raw_train_dataset = Subset(raw_dataset, train_indices)
    raw_test_dataset = Subset(raw_dataset, test_indices)

    # Wrap with transform
    train_dataset = MyDataset(raw_train_dataset, get_train_transform())
    test_dataset = MyDataset(raw_test_dataset, get_test_transform())

    return train_dataset, test_dataset