
from sift_config import *
from sift_feature_extraction import *
from core.process_dataset import *
from core.evaluations import *
import cv2
import numpy as np
import matplotlib.pyplot as plt
from joblib import dump, load

FEATURE_DIR = "cached_features"
os.makedirs(FEATURE_DIR, exist_ok=True)

def build_classifier(classifier_type=None):
    classifier_type = classifier_type or CLASSIFIER_TYPE
    if classifier_type == "svm":
        return SVC(kernel=SVM_KERNEL, random_state=RANDOM_STATE)
    elif classifier_type == "knn":
        return KNeighborsClassifier(n_neighbors=KNN_K)
    else:
        raise ValueError(f"Unsupported classifier type: {classifier_type}")

def train_sift_model(train_images, train_labels, classifier_type=None):
    classifier_type = classifier_type or CLASSIFIER_TYPE
    train_feat_path = os.path.join(FEATURE_DIR, "sift_train_features.npy")
    train_label_path = os.path.join(FEATURE_DIR, "sift_train_labels.npy")
    kmeans_path = os.path.join(FEATURE_DIR, "sift_kmeans.joblib")
    if os.path.exists(train_feat_path) and os.path.exists(train_label_path) and os.path.exists(kmeans_path):
        print("  Load cached training features...")
        X_train = np.load(train_feat_path)
        y_train = np.load(train_label_path)
        kmeans = load(kmeans_path)
    else:
        print("Extract SIFT features from training set...")
        train_desc = extract_color_sift_features(train_images)
        kmeans = create_visual_vocabulary(train_desc)

        X_train = extract_bow_features(train_desc, kmeans)
        y_train = np.array(train_labels)
        np.save(train_feat_path, X_train)
        np.save(train_label_path, y_train)
        dump(kmeans, kmeans_path)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)

    clf = build_classifier(classifier_type)
    clf.fit(X_train, y_train)

    if SAVE_MODEL:
        os.makedirs(MODEL_DIR, exist_ok=True)
        joblib.dump(kmeans, os.path.join(MODEL_DIR, f"kmeans_{classifier_type}.pkl"))
        joblib.dump(scaler, os.path.join(MODEL_DIR, f"scaler_{classifier_type}.pkl"))
        joblib.dump(clf, os.path.join(MODEL_DIR, f"classifier_{classifier_type}.pkl"))
        print(f"  The model has been saved as classifier_{classifier_type}.pkl")

    return kmeans, scaler, clf

def load_sift_model(classifier_type=None):
    classifier_type = classifier_type or CLASSIFIER_TYPE
    kmeans = joblib.load(os.path.join(MODEL_DIR, f"kmeans_{classifier_type}.pkl"))
    scaler = joblib.load(os.path.join(MODEL_DIR, f"scaler_{classifier_type}.pkl"))
    clf = joblib.load(os.path.join(MODEL_DIR, f"classifier_{classifier_type}.pkl"))
    print(f"  Model loading completed:classifier_{classifier_type}.pkl")
    return kmeans, scaler, clf

def test_sift_model(test_images, test_labels, kmeans, scaler, clf, classes):
    test_feat_path = os.path.join(FEATURE_DIR, "sift_test_features.npy")
    test_label_path = os.path.join(FEATURE_DIR, "sift_test_labels.npy")

    if os.path.exists(test_feat_path) and os.path.exists(test_label_path):
        print("  Loading cached test features...")
        X_test = np.load(test_feat_path)
        y_test = np.load(test_label_path)
    else:
        print("  Extracting SIFT features from the test set...")
        test_desc = extract_color_sift_features(test_images)
        X_test = extract_bow_features(test_desc, kmeans)
        y_test = np.array(test_labels)
        np.save(test_feat_path, X_test)
        np.save(test_label_path, y_test)

    X_test = scaler.transform(X_test)
    y_pred = clf.predict(X_test)
    # print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
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





def train_imbalance_model(train_images, train_labels, classifier_type=None):
    classifier_type = classifier_type or CLASSIFIER_TYPE
    kmeans_path = os.path.join(FEATURE_DIR, "sift_kmeans.joblib")

    print("Extract SIFT features from training set...")
    train_desc = extract_color_sift_features(train_images)
    kmeans = create_visual_vocabulary(train_desc)

    X_train = extract_bow_features(train_desc, kmeans)
    y_train = np.array(train_labels)
    dump(kmeans, kmeans_path)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)

    clf = build_classifier(classifier_type)
    clf.fit(X_train, y_train)

    return kmeans, scaler, clf

def test_sift_model_imbalance(test_images, test_labels, kmeans, scaler, clf, classes):
    """
    Test functions dedicated to unbalanced experiments, skipping the cache mechanism and ensuring the use of current kmeans.
    """
    print("  Extract test set SIFT features (without using cache)...")
    test_desc = extract_color_sift_features(test_images)
    X_test = extract_bow_features(test_desc, kmeans)
    y_test = np.array(test_labels)

    
    X_test = scaler.transform(X_test)
    y_pred = clf.predict(X_test)

    return y_test, y_pred, classes