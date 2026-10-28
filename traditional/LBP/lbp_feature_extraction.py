import numpy as np
from skimage.feature import local_binary_pattern

RADIUS = 2
N_POINTS = 8 * RADIUS
METHOD = 'uniform'

def extract_lbp_features(images):
    features = []
    for img in images:
        if len(img.shape) == 3: 
            from cv2 import cvtColor, COLOR_BGR2GRAY
            img = cvtColor(img, COLOR_BGR2GRAY)
        lbp = local_binary_pattern(img, N_POINTS, RADIUS, METHOD)
        hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, N_POINTS + 3), range=(0, N_POINTS + 2))
        hist = hist.astype("float")
        hist /= (hist.sum() + 1e-7)
        features.append(hist)
    return np.array(features)