from torchvision import transforms
from utils.image_utils import add_black_patch, add_noise, blur_image

# Add occlusion (black patch)
def get_patch_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Lambda(lambda img: add_black_patch(img)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
    ])

# Add blur
def get_blur_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Lambda(lambda img: blur_image(img)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
    ])

# Add noise
def get_noise_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Lambda(lambda img: add_noise(img)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
    ])
