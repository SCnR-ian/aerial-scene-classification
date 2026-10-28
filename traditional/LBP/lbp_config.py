import os
from core.process_dataset import sift_split_data  
from sklearn.metrics import *
import matplotlib.pyplot as plt
import seaborn as sns

RANDOM_STATE = 42
SAMPLE_RATIO = 1.0
TEST_SIZE = 0.2
CLASSIFIER_TYPE = "svm"
SVM_KERNEL = "linear"
KNN_K = 5
SAVE_MODEL = True
LOAD_MODEL = False
LOAD_COLOR_IMAGES = False
DATA_ROOT = "/9517/dataset/Aerial_Landscapes"
MODEL_DIR = "saved_models"

def get_dataset():
    return sift_split_data(
        data_root=DATA_ROOT,
        sample_ratio=SAMPLE_RATIO,
        test_ratio=TEST_SIZE,
        shuffle_seed=RANDOM_STATE,
        load_color_images=LOAD_COLOR_IMAGES
    )