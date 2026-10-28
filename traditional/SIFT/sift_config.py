import cv2
import numpy as np
import os
from glob import glob
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import MiniBatchKMeans
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


N_CLUSTERS = 100
MAX_SAMPLES = 10000
TEST_SIZE = 0.2
RANDOM_STATE = 42
SAMPLE_RATIO = 1.0
LOAD_COLOR_IMAGES = True
SVM_KERNEL = "rbf"


DATA_ROOT = "/9517/dataset/Aerial_Landscapes"

SAVE_MODEL = True
LOAD_MODEL = False  

MODEL_DIR = "saved_models"
CLASSIFIER_TYPE = "svm"  
KNN_K = 5               

KMEANS_PATH = os.path.join(MODEL_DIR, f"kmeans_{CLASSIFIER_TYPE}.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, f"scaler_{CLASSIFIER_TYPE}.pkl")
CLASSIFIER_PATH = os.path.join(MODEL_DIR, f"classifier_{CLASSIFIER_TYPE}.pkl")



def get_dataset():
    from core.process_dataset import sift_split_data
    return sift_split_data(
        data_root=DATA_ROOT,
        sample_ratio=SAMPLE_RATIO,
        test_ratio=TEST_SIZE,
        shuffle_seed=RANDOM_STATE,
        load_color_images=LOAD_COLOR_IMAGES
    )


if __name__ == '__main__':
    for k, v in globals().items():
        if not k.startswith('_') and not callable(v):
            print(f"{k}: {v}")