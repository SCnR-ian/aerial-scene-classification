import os
import torch
import shutil
from torchvision import datasets
from torch.utils.data import DataLoader
from image_preprocess import prepare_datasets
from timm import create_model
from tqdm import tqdm

# Set parameters
DATASET_PATH = 'dataset/aerial_landscapes'
MODEL_PATH = 'best_model.pth'
NUM_CLASSES = 15
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create output directories
os.makedirs('success_images', exist_ok=True)
os.makedirs('failure_images', exist_ok=True)

# Load the model
model = create_model('seresnet50', pretrained=False, num_classes=NUM_CLASSES)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval().to(DEVICE)

# Load data (transforms already included)
_, test_dataset = prepare_datasets(DATASET_PATH, seed=18)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Get list of original image paths
# Note: image_paths can only be tracked after subsetting the original ImageFolder
raw_dataset = datasets.ImageFolder(DATASET_PATH)
image_paths = [raw_dataset.samples[i][0] for i in test_dataset.subset.indices]

success, failure = [], []

for i, (img, label) in enumerate(tqdm(test_loader)):
    img, label = img.to(DEVICE), label.to(DEVICE)
    with torch.no_grad():
        output = model(img)
        pred = torch.argmax(output, dim=1)

    src_path = image_paths[i]
    base_name = os.path.basename(src_path)
    pred_label = test_dataset.subset.dataset.classes[pred.item()]
    true_label = test_dataset.subset.dataset.classes[label.item()]

    # Construct new filename: true_pred_filename.jpg
    new_name = f"{true_label}_pred_{pred_label}_{base_name}"
    dst_path = os.path.join('success_images' if pred == label else 'failure_images', new_name)
    shutil.copy(src_path, dst_path)

    if pred == label:
        success.append((src_path, true_label, pred_label))
    else:
        failure.append((src_path, true_label, pred_label))

print(f"\nCorrectly classified: {len(success)} images")
print(f"Misclassified: {len(failure)} images")
print("Images have been saved to the `success_images/` and `failure_images/` folders, respectively")
