import numpy as np
import os
import joblib
from sklearn.preprocessing import StandardScaler
from lbp_feature_extraction import extract_lbp_features
from lbp_config import *
import cv2
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

FEATURE_DIR = "cached_features_lbp"
os.makedirs(FEATURE_DIR, exist_ok=True)

def build_classifier(classifier_type=None):
    classifier_type = classifier_type or CLASSIFIER_TYPE
    if classifier_type == "svm":
        return SVC(kernel=SVM_KERNEL, random_state=RANDOM_STATE)
    elif classifier_type == "knn":
        return KNeighborsClassifier(n_neighbors=KNN_K)
    else:
        raise ValueError(f"Unsupported classifier type: {classifier_type}")

def train_lbp_model(train_images, train_labels, classifier_type=None):
    classifier_type = classifier_type or CLASSIFIER_TYPE
    os.makedirs(FEATURE_DIR, exist_ok=True)
    feature_path = os.path.join(FEATURE_DIR, f"lbp_train_features.npy")
    label_path = os.path.join(FEATURE_DIR, f"lbp_train_labels.npy")

    if os.path.exists(feature_path) and os.path.exists(label_path):
        print("  Load cached LBP features...")
        X_train = np.load(feature_path)
        y_train = np.load(label_path)
    else:
        print("  Extract training set LBP features...")
        X_train = extract_lbp_features(train_images)
        y_train = np.array(train_labels)
        np.save(feature_path, X_train)
        np.save(label_path, y_train)
        print(f"  Save features to {feature_path}")

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)

    clf = build_classifier(classifier_type)
    clf.fit(X_train, y_train)

    if SAVE_MODEL:
        os.makedirs(MODEL_DIR, exist_ok=True)
        joblib.dump(scaler, os.path.join(MODEL_DIR, f"lbp_scaler_{classifier_type}.pkl"))
        joblib.dump(clf, os.path.join(MODEL_DIR, f"lbp_classifier_{classifier_type}.pkl"))
        print(f"  The model has been saved as classifier_{classifier_type}.pkl")
    return scaler, clf

def load_lbp_model(classifier_type=None):
    classifier_type = classifier_type or CLASSIFIER_TYPE
    scaler = joblib.load(os.path.join(MODEL_DIR, f"lbp_scaler_{classifier_type}.pkl"))
    clf = joblib.load(os.path.join(MODEL_DIR, f"lbp_classifier_{classifier_type}.pkl"))
    print(f"  Model loading completed: classifier_{classifier_type}.pkl")
    return scaler, clf

def test_lbp_model(test_images, test_labels, scaler, clf, classes):
    os.makedirs(FEATURE_DIR, exist_ok=True)
    feature_path = os.path.join(FEATURE_DIR, "lbp_test_features.npy")
    label_path = os.path.join(FEATURE_DIR, "lbp_test_labels.npy")

    if os.path.exists(feature_path) and os.path.exists(label_path):
        print("  Load cached LBP test features...")
        X_test = np.load(feature_path)
        y_test = np.load(label_path)
    else:
        print("  Extract test set LBP features...")
        X_test = extract_lbp_features(test_images)
        y_test = np.array(test_labels)
        np.save(feature_path, X_test)
        np.save(label_path, y_test)

    X_test = scaler.transform(X_test)
    y_pred = clf.predict(X_test)
    return y_test, y_pred, classes

def apply_perturbation(images, mode="noise", strength=0.04, clip=False, visualize=True):
    """
    Apply perturbation to the input image set (supports visualization)
    Args:
    mode: "noise", "blur", "occlusion"
    strength: perturbation strength (noise is the standard deviation ratio)
    clip: whether to clip to prevent overflow
    visualize: whether to visualize the before and after image comparison

    """
    perturbed = []

    for i, img in enumerate(images):
        if mode == "noise":
            noise = np.random.normal(0, strength * 255, img.shape)
            perturbed_img = img + noise
            if clip:
                perturbed_img = np.clip(perturbed_img, 0, 255)
            perturbed_img = perturbed_img.astype(np.uint8)

        elif mode == "blur":
            perturbed_img = cv2.GaussianBlur(img, (5, 5), strength * 10)

        elif mode == "occlusion":
            perturbed_img = img.copy()
            h, w = img.shape[:2]
            cv2.rectangle(perturbed_img, (w//4, h//4), (w//2, h//2), (0, 0, 0), -1)

        else:
            perturbed_img = img

        perturbed.append(perturbed_img)

        if visualize and i < 1:
            fig, axs = plt.subplots(1, 2, figsize=(8, 4))
            axs[0].imshow(img, cmap="gray" if len(img.shape)==2 else None)
            axs[0].set_title("Original")
            axs[0].axis("off")

            axs[1].imshow(perturbed_img, cmap="gray" if len(img.shape)==2 else None)
            axs[1].set_title(f"Perturbed: {mode}")
            axs[1].axis("off")
            plt.tight_layout()
            plt.show()

    return perturbed


from collections import defaultdict
from sklearn.utils import resample

def simulate_imbalance_and_resample(images, labels, 
                                    imbalance_ratio: dict = None,
                                    resample_method: str = None,
                                    random_state: int = 42):
    """
    Simulate class imbalance and perform undersampling/oversampling

    Args:
    images: image list
    labels: label list
    imbalance_ratio: dict, for example {0: 1.0, 1: 0.5, 2: 0.3}, the rest of the classes default to 1.0
    resample_method: "oversample", "undersample" or None
    random_state: ensure reproducibility

    Returns:
    new_images, new_labels
    """
    if imbalance_ratio is None:
        imbalance_ratio = {}

    label_to_data = defaultdict(list)
    for img, label in zip(images, labels):
        label_to_data[label].append(img)

    new_images, new_labels = [], []
    for label, imgs in label_to_data.items():
        ratio = imbalance_ratio.get(label, 1.0)
        n_select = max(1, int(len(imgs) * ratio))
        selected_imgs = resample(imgs, n_samples=n_select, random_state=random_state)
        new_images.extend(selected_imgs)
        new_labels.extend([label] * n_select)

    if resample_method == "oversample":
        max_count = max([new_labels.count(l) for l in set(new_labels)])
        resampled_images, resampled_labels = [], []
        for label in set(new_labels):
            imgs = [img for img, l in zip(new_images, new_labels) if l == label]
            imgs_resampled = resample(imgs, n_samples=max_count, random_state=random_state)
            resampled_images.extend(imgs_resampled)
            resampled_labels.extend([label] * max_count)
        return resampled_images, resampled_labels

    elif resample_method == "undersample":
        min_count = min([new_labels.count(l) for l in set(new_labels)])
        resampled_images, resampled_labels = [], []
        for label in set(new_labels):
            imgs = [img for img, l in zip(new_images, new_labels) if l == label]
            imgs_resampled = resample(imgs, n_samples=min_count, random_state=random_state)
            resampled_images.extend(imgs_resampled)
            resampled_labels.extend([label] * min_count)
        return resampled_images, resampled_labels

    return new_images, new_labels

def train_lbp_imbalanced_model(train_images, train_labels, classifier_type=None):
    classifier_type = classifier_type or CLASSIFIER_TYPE
    print("  Extract training set LBP features...")
    X_train = extract_lbp_features(train_images)
    y_train = np.array(train_labels)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)

    clf = build_classifier(classifier_type)
    clf.fit(X_train, y_train)

    return scaler, clf


def test_lbp_model_imbalance(test_images, test_labels, scaler, clf, classes):

    print("  Extract test set LBP features...")
    X_test = extract_lbp_features(test_images)
    y_test = np.array(test_labels)
    X_test = scaler.transform(X_test)
    y_pred = clf.predict(X_test)
    return y_test, y_pred, classes