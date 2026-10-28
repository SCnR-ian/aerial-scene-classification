import cv2
import numpy as np
from PIL import Image

# Add black patch
def add_black_patch(img, top_left=(60, 60), size=(50, 50)):
    img = np.array(img).copy()
    x, y = top_left
    w, h = size
    img[y:y+h, x:x+w, :] = 0
    return Image.fromarray(img)

# Blur image
def blur_image(img, ksize=11):
    img = np.array(img)
    blurred = cv2.GaussianBlur(img, (ksize, ksize), 0)
    return Image.fromarray(blurred)

# Add Gaussian noise
def add_noise(img, level=20): 
    img = np.array(img).astype(np.float32)
    noise = np.random.normal(0, level, img.shape).astype(np.float32)
    noisy = np.clip(img + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy)

