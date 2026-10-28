import torch
import torch.nn as nn
from torch.utils.data import WeightedRandomSampler
from torchvision import transforms
import numpy as np

# Generate weights based on the number of samples per class and return a weighted CrossEntropyLoss
def get_class_weighted_loss(class_counts, device):
    weights = [1.0 / c if c > 0 else 0.0 for c in class_counts]
    weights = torch.tensor(weights)
    weights = weights / weights.sum()  # Normalize
    return nn.CrossEntropyLoss(weight=weights.to(device))

# Given the labels of all training samples, return a WeightedRandomSampler
def get_weighted_sampler(labels):
    class_sample_counts = np.bincount(labels)
    weights = 1. / class_sample_counts
    sample_weights = [weights[label] for label in labels]
    return WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

# Return a stronger augmentation strategy for minority classes
def get_aug_transform():
    return transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),
        transforms.RandomRotation(30),
        transforms.ColorJitter(0.5, 0.5, 0.5),
        transforms.RandomHorizontalFlip(),
        transforms.RandomGrayscale(p=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
    ])
