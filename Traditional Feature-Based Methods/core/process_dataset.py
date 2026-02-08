import os
import random
import numpy as np
import cv2
from glob import glob


def sift_split_data(data_root: str, test_ratio: float = 0.2, sample_ratio: float = 1.0,
                     shuffle_seed: int = 0, load_color_images: bool = True):

    assert os.path.exists(data_root), f"Data root {data_root} does not exist."
    random.seed(shuffle_seed)

    class_names = [d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d))]
    class_names.sort()

    train_images, train_labels = [], []
    test_images, test_labels = [], []

    for class_index, class_name in enumerate(class_names):
        class_dir = os.path.join(data_root, class_name)
        img_paths = glob(os.path.join(class_dir, '*.jpg'))

        random.shuffle(img_paths)
        sample_num = int(len(img_paths) * sample_ratio)
        img_paths = img_paths[:sample_num]

        split_idx = int(sample_num * (1 - test_ratio))
        train_paths = img_paths[:split_idx]
        test_paths = img_paths[split_idx:]

        for path in train_paths:
            img = cv2.imread(path) if load_color_images else cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                train_images.append(img)
                train_labels.append(class_index)

        for path in test_paths:
            img = cv2.imread(path) if load_color_images else cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                test_images.append(img)
                test_labels.append(class_index)

    print(f"Total: {len(train_images)} training images, {len(test_images)} testing images")
    return train_images, train_labels, test_images, test_labels, class_names


def load_imbalanced_dataset(data_root, test_ratio=0.2, sample_ratio=1.0, gray=False, per_class_sample_ratio=None):
    class_names = sorted(os.listdir(data_root))
    train_images, train_labels = [], []
    test_images, test_labels = [], []

    for class_index, class_name in enumerate(class_names):
        class_dir = os.path.join(data_root, class_name)
        img_paths = glob(os.path.join(class_dir, '*.jpg'))
        random.shuffle(img_paths)

        ratio = per_class_sample_ratio.get(class_index, sample_ratio) if per_class_sample_ratio else sample_ratio
        sample_num = int(len(img_paths) * ratio)
        img_paths = img_paths[:sample_num]

        split_idx = int(sample_num * (1 - test_ratio))

        for path in img_paths[:split_idx]:
            img = cv2.imread(path)
            if gray and img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            train_images.append(img)
            train_labels.append(class_index)

        for path in img_paths[split_idx:]:
            img = cv2.imread(path)
            if gray and img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            test_images.append(img)
            test_labels.append(class_index)

    return train_images, train_labels, test_images, test_labels, class_names



