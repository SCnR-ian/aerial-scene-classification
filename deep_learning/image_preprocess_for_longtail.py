from collections import defaultdict
import numpy as np
from torchvision import datasets
from torch.utils.data import Subset, Dataset
import torch
import random

from image_preprocess import MyDataset, get_test_transform, get_train_transform
from utils.imbalanced_utils import get_aug_transform

class MyLongtailDataset(Dataset):
    def __init__(self, subset, minority_classes, normal_transform, minority_transform):
        self.subset = subset
        self.minority_classes = set(minority_classes)
        self.normal_transform = normal_transform
        self.minority_transform = minority_transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        img, label = self.subset[idx]
        transform = self.minority_transform if label in self.minority_classes else self.normal_transform
        return transform(img), label


# Simulate long-tailed distribution with exponential decay
def prepare_longtail_datasets(dataset_path, train_ratio=0.8, seed=42, dataset_aug=True, 
                                tail_factor=0.82):

    raw_dataset = datasets.ImageFolder(dataset_path)
    class_indices = defaultdict(list)

    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    for idx, (_, label) in enumerate(raw_dataset):
        class_indices[label].append(idx)
    # Simulate number of images per class (exponential decay)
    simulated_counts = [int(800 * (tail_factor ** i)) for i in range(len(class_indices))]
    print("Simulated image count per class (long-tail):", simulated_counts)

    train_indices, test_indices = [], []

    for label, indices in class_indices.items():
        random.shuffle(indices)

        n_total = min(len(indices), simulated_counts[label])
        n_train = int(n_total * train_ratio)
        sampled = indices[:n_total]

        train_indices += sampled[:n_train]
        test_indices += sampled[n_train:]

    raw_train_dataset = Subset(raw_dataset, train_indices)
    raw_test_dataset = Subset(raw_dataset, test_indices)

    minority_classes = [i for i, count in enumerate(simulated_counts) if count < 100]

    if dataset_aug:
        train_dataset = MyLongtailDataset(raw_train_dataset, minority_classes, get_train_transform(), get_aug_transform())
    else:
        train_dataset = MyDataset(raw_train_dataset, get_test_transform()) #Use test transform: only resize the image without other transformations
    test_dataset = MyDataset(raw_test_dataset, get_test_transform())

    return train_dataset, test_dataset, simulated_counts