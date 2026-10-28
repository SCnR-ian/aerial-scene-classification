# Aerial Scene Classification

Classifying aerial photos into 15 land-use types (farmland, airport, beach, city, forest and so on), comparing traditional computer vision with deep learning. Team project for UNSW COMP9517 (Computer Vision).

**My part:** the deep learning side, everything in [`deep_learning/`](deep_learning/). I fine-tuned the three CNNs, ran the long-tail experiments and built the robustness and Grad-CAM tests. Teammates built the SIFT and LBP baselines in [`traditional/`](traditional/).

## Results

The dataset has 12,000 images (800 per class), split 80/20 into training and test. All numbers are test-set accuracy on 2,400 images.

### Deep learning vs traditional features

| Approach | Model | Accuracy |
| --- | --- | --- |
| Traditional | LBP texture features + SVM | 60.1% |
| Traditional | SIFT bag of words + SVM | 67.5% |
| Deep learning | ResNet-50 (fine-tuned) | 98.5% |
| Deep learning | SE-ResNet-50 (fine-tuned) | 98.6% |
| Deep learning | **EfficientNet-B0 (fine-tuned)** | **98.8%** |

Fine-tuned CNNs beat the best traditional pipeline by over 30 percentage points.

### When some classes are rare (long tail)

Real-world data is rarely balanced, so I rebuilt the training set so each class had 18% fewer images than the one before it (800 images for the largest class down to under 50 for the smallest), then trained SE-ResNet-50 on it and tested on the normal, balanced test set.

| Fix for the imbalance | Accuracy |
| --- | --- |
| None | 94.9% |
| Class-weighted loss | 96.4% |
| Resampling rare classes more often | 96.7% |
| Extra augmentation for rare classes | 97.0% |
| **Class-weighted loss + resampling** | **97.2%** |

### How the models hold up on damaged images

| Test images with... | ResNet-50 | SE-ResNet-50 | EfficientNet-B0 |
| --- | --- | --- | --- |
| No damage | 98.5% | 98.6% | 98.8% |
| A black patch covering part of the image | 97.9% | 97.8% | 97.7% |
| Gaussian blur | 88.2% | 88.9% | 82.7% |
| Gaussian noise | 71.7% | **80.9%** | 13.6% |

**Key finding:** the most accurate model on clean images was the least robust. EfficientNet-B0 fell to 13.6% under noise, while SE-ResNet-50 kept 80.9%. Picking a model on clean accuracy alone would have chosen the wrong one for noisy real-world images. [`gradcam_perturb.ipynb`](deep_learning/gradcam_perturb.ipynb) shows Grad-CAM heatmaps of where each model looks, before and after the damage.

## Repository layout

```
deep_learning/        my part
  train.py            fine-tune ResNet-50, EfficientNet-B0 or SE-ResNet-50
  train_longtail.py   same, on the long-tail dataset, with weighting and resampling options
  evaluate_*.ipynb    test-set results, including the damaged-image tests
  gradcam_perturb.ipynb
  utils/              data loading, image damage functions, imbalance helpers
traditional/          SIFT and LBP feature pipelines with SVM and KNN classifiers
```

Each folder has its own README with setup steps.

## Tech

Python, PyTorch, torchvision, timm, OpenCV, scikit-learn, pytorch-grad-cam
