## Project
This repository contains python notebooks for training, long‑tail training, evaluation, and Grad‑CAM visualization of `RESNET` `EFFICIENTNET` `SENet` models.

## Common Setup
You can install all dependencies from the requirements file:
```
pip install -r requirements.txt
```
---

#### 1. Standard Training (`train.py`)
At the top of `train.py` you’ll find a block of **uppercase** variables. You can adjust these before you run training. 

Once you’ve adjusted the configuration at the top of train.py, start training with:
```
python train.py
```
- Each epoch prints out training loss and test accuracy.

- Checkpoints for every epoch are saved into the directory you set in CHECKPOINT_DIR.

- When a new highest test accuracy is reached, the script also writes best_model.pth in the working directory.

#### 2. Simulating Long‑Tail Training (`train_longtail.py`)
For the train_longtail.py, everything is configured exactly as in train.py except for two additional flags at the top:

- `USE_CLASS_WEIGHT`
Set to True to apply a class‑weighted CrossEntropyLoss, where each class is weighted inversely to its sample count.

- `USE_RESAMPLING`
Set to True to enable a weighted sampler in the DataLoader, so that examples are drawn according to class weights.

Once you’ve edited those two flags (and any of the usual settings), simply run:

```
python train_longtail.py
```

- If `USE_CLASS_WEIGHT=True`, the loss function will weight each class inversely to its frequency.

- If `USE_RESAMPLING=True`, the training DataLoader will sample examples according to those same weights.

You’ll get one checkpoint per epoch in the folder you set for CHECKPOINT_DIR.

When a new highest test accuracy is reached, the script saves the best long‑tail model as best_model_longtail.pth.

#### 3. Model Evaluation Notebook (`evaluate.ipynb`)
This notebook generates detailed classification reports and confusion matrices for:

- The base model and three perturbations: image blur, add noise, and add black patch
- Long‑tail variants: base, resampling, reweighting, both  
- Bar charts comparing accuracy and F1-score across each group

Just open the notebook and run all cells.

#### 4. Inference & Result Analysis (`analyze_result.py`)
This script runs inference on the test set and copies each image into **success_images/** or **failure_images/**, renaming files to include true and predicted labels.

At the top of **`analyze_result.py`**, edit:

- `DATASET_PATH` — path to the dataset folder (default: `dataset/aerial_landscapes`)  
- `MODEL_PATH` — path to the saved model weights (default: `best_model.pth`)  
Run:
```
python analyze_result.py
```

#### 5. Grad‑CAM Perturbation Notebook (`gradcam_perturb.ipynb`)
This notebook visualizes Grad‑CAM attention maps for a single image under different perturbations.
After edit the `IMG_PATH` and run all cells, this notebook will produce:

- The original image and its Grad‑CAM map with predicted class
- Perturbed versions (blur, occlusion, noise) and their Grad‑CAM maps with predicted classes

Use this to inspect whether the model still focuses on the right regions under distortion.
