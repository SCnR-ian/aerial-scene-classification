from sift_feature_extraction import extract_color_sift_features, extract_bow_features
from sift_utils import *
import numpy as np
from sklearn.metrics import accuracy_score, f1_score



def test_robustness_with_saved_model(test_images, test_labels, classes, classifier_type="svm"):
    kmeans, scaler, clf = load_sift_model(classifier_type)
    
    modes = ["original", "noise", "blur", "occlusion"]
    acc_list, f1_list = [], []

    for mode in modes:
        print(f"💥 Testing under perturbation: {mode}")
        perturbed_images = apply_perturbation(test_images, mode=mode)
        test_desc = extract_color_sift_features(perturbed_images)
        X_test = extract_bow_features(test_desc, kmeans)
        X_test = scaler.transform(X_test)
        y_pred = clf.predict(X_test)
        acc = accuracy_score(test_labels, y_pred) 
        f1 = f1_score(test_labels, y_pred, average="macro")
        acc_list.append(acc)
        f1_list.append(f1)

    return modes, acc_list, f1_list