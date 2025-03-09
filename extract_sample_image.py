import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os

# CIFAR-10 class names
CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

# Create a directory to save sample images
os.makedirs('sample_images', exist_ok=True)

# Load CIFAR-10 dataset
dataset = datasets.CIFAR10(root='./data', train=False, download=True)

# Extract and save some sample images (one per class)
for class_idx in range(10):
    # Find an image from this class
    for idx, (img, label) in enumerate(dataset):
        if label == class_idx:
            # Convert from PIL to numpy array
            img_array = np.array(img)
            
            # Save as a regular image file
            img_pil = Image.fromarray(img_array)
            save_path = f'sample_images/class_{class_idx}_{CIFAR10_CLASSES[class_idx]}.png'
            img_pil.save(save_path)
            
            print(f"Saved {save_path}")
            break  # Move to next class after finding one example

print("\nSample images extracted from CIFAR-10 dataset.")
print("You can now test the model using the image_predictor.py script:")
print("Example: python image_predictor.py --image sample_images/class_0_airplane.png")
