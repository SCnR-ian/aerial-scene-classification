# Remote Sensing Image Classification using SIFT and LBP

This project compares traditional feature extraction methods (SIFT and LBP) with classical classifiers (SVM and KNN) on a 15-class aerial landscape dataset. It includes baseline accuracy evaluation, robustness testing, and imbalanced class simulation.

---

##  Features

- SIFT + BoW feature extraction with KMeans clustering
- LBP histogram extraction for texture-based classification
- SVM / KNN classification for both methods
- Evaluation metrics: Accuracy, Precision, Recall, F1-score
- Confusion matrix & bar plots
- Robustness tests (Gaussian noise, blur, occlusion)
- Imbalanced class simulation with resampling strategies

---

##  Requirements

Install all required libraries:

```bash
pip install opencv-python scikit-learn scikit-image matplotlib numpy joblib
 ```
---

## Data Setup

### 1.Download and place your dataset inside the root project directory (e.g. data/remote_sensing_images/).
### 2.Make sure the folder structure is:
```bash
project_root/
├── data/
│   └── remote_sensing_images/
│       ├── class_1/
│       ├── class_2/
│       └── ...
 ```
### 3.Modify the dataset path in the corresponding config file:
For SIFT: SIFT/sift_config.py  
For LBP: LBP/lbp_config.py

Example:  
DATA_ROOT = "../data/remote_sensing_images"

---

## How to Use
### 1. Modify Configuration File

Before running experiments, update the corresponding config file:

- For SIFT: edit `SIFT/sift_config.py`
- For LBP: edit `LBP/lbp_config.py`

Example configuration options:

```python
# Example: SIFT/sift_config.py

DATA_ROOT = "../data/remote_sensing_images"  # Path to your dataset
CLASSIFIER_TYPE = "svm"                      # Choose from "svm" or "knn"
SAMPLE_RATIO = 1.0                           # Proportion of samples to use
SAVE_MODEL = True                            # Whether to save the trained model
```
### 2. Run the Notebook
 - Run SIFT/sift_compare_and_analyze.ipynb to test SIFT + SVM/KNN  
 - Run LBP/lbp_compare.ipynb to test LBP + SVM/KNN
   
The notebooks will automatically read config settings and execute:  
 - Feature extraction (with caching to avoid recomputation)  
 - Model training and evaluation  
 - Confusion matrix and bar chart visualization  
 - Robustness testing (noise, blur, occlusion)  
 - Imbalanced class simulation and resampling experiments  
   
No need to modify source code — just adjust the config and run the notebook.

## References
OpenCV SIFT API: https://docs.opencv.org/

scikit-learn: https://scikit-learn.org/

scikit-image (LBP): https://scikit-image.org/

Bag of Words + KMeans: https://www.learnopencv.com/image-classification-using-bag-of-visual-words-models-in-opencv/

Resampling Utility: https://scikit-learn.org/stable/modules/generated/sklearn.utils.resample.html
