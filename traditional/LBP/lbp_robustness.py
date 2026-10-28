from lbp_feature_extraction import extract_lbp_features
import numpy as np
from lbp_utils import *
from sklearn.metrics import accuracy_score, f1_score


def test_robustness_with_saved_model(test_images, test_labels, classes, 
                                            classifier_type="svm"):
    '''
    Use the saved model to evaluate robustness. LBP does not use BoW

    Args:
    test_images: image list
    test_labels: label list
    classes: category name
    classifier_type: "svm"/"knn"
    feature_type: "sift"/"lbp"

    Returns:
    modes, accuracy_list, f1_list
    '''

    scaler, clf = load_lbp_model(classifier_type)

    modes = ["original", "noise", "blur", "occlusion"]
    acc_list, f1_list = [], []

    for mode in modes:
        print(f"  Testing under perturbation: {mode}")
        perturbed_images = apply_perturbation(test_images, mode=mode)

 
        X_test = extract_lbp_features(perturbed_images)  

        X_test = scaler.transform(X_test)
        y_pred = clf.predict(X_test)

        acc = accuracy_score(test_labels, y_pred) 
        f1 = f1_score(test_labels, y_pred, average="macro")
        acc_list.append(acc)
        f1_list.append(f1)

    return modes, acc_list, f1_list



