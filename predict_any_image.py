import os
import sys
import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import numpy as np

from models.vgg import VGG
from utils.configs import Config

# CIFAR-10 class names
CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

def load_model(model_path=None):
    """
    Load a VGG model, either from a saved checkpoint or create a new one.

    If a model path is provided, load the saved weights from the checkpoint.
    Otherwise, create a new VGG11 model with batch normalization.

    Args:
        model_path: Path to a saved model checkpoint (optional)

    Returns:
        model: Loaded or created VGG model
        device: Device where the model is placed
    """
    # Create a VGG11 model with batch normalization
    config = Config.vgg11(batch_norm=True, num_classes=10)
    model = VGG(config)
    
    # If a model path is provided, load the saved weights
    if model_path and os.path.exists(model_path):
        print(f"Loading model from {model_path}")
        checkpoint = torch.load(model_path)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
    else:
        print("Using a new VGG11 model (not trained)")
    
    # Move model to the appropriate device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()  # Set to evaluation mode
    
    return model, device

def preprocess_image(image_path, device):
    """Process an image to be suitable for the VGG model"""
    try:
        # Check if file exists
        if not os.path.exists(image_path):
            print(f"Error: Image file {image_path} not found.")
            return None, None
            
        # VGG preprocessing: resize to 224x224 and normalize with ImageNet stats
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Open and preprocess the image
        image = Image.open(image_path).convert('RGB')
        original_image = image.copy()
        
        # Apply transformations
        image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
        image_tensor = image_tensor.to(device)
        
        return image_tensor, original_image
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None, None

def make_prediction(model, image_tensor):
    """Make a prediction with the model"""
    if image_tensor is None:
        return None, None
        
    with torch.no_grad():
        outputs = model(image_tensor)
        
    # Get probabilities
    probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
    
    # Get top predictions
    top_probs, top_indices = torch.topk(probabilities, 3)
    
    return top_probs.cpu().numpy(), top_indices.cpu().numpy()

def visualize_predictions(images, predictions, save_dir=None):
    """Display multiple images with their prediction results"""
    # Calculate grid size
    n_images = len(images)
    cols = min(3, n_images)
    rows = (n_images + cols - 1) // cols
    
    # Create figure with one row per image
    plt.figure(figsize=(15, 5 * rows))
    
    for i, (img_path, (image, probs, indices)) in enumerate(zip(images, predictions)):
        # Skip if prediction failed
        if image is None or probs is None:
            continue
            
        # Get filename for title
        filename = Path(img_path).name
        
        # Display original image
        plt.subplot(rows, 2*cols, 2*i+1)
        plt.imshow(image)
        plt.title(f"Image: {filename}")
        plt.axis('off')
        
        # Display top 3 predictions as a bar chart
        plt.subplot(rows, 2*cols, 2*i+2)
        
        # Create colors based on probability (higher probability = deeper color)
        colors = plt.cm.YlOrRd(probs)
        
        # Get class names
        class_names = [CIFAR10_CLASSES[idx] for idx in indices]
        y_pos = np.arange(len(class_names))
        
        # Draw horizontal bars
        plt.barh(y_pos, probs * 100, color=colors)
        plt.yticks(y_pos, class_names)
        plt.xlabel('Probability (%)')
        plt.title('Predictions')
        
        # Add percentage text
        for j, prob in enumerate(probs):
            plt.text(prob * 100 + 1, j, f"{prob*100:.1f}%")
    
    plt.tight_layout()
    
    # Save the visualization if requested
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        output_path = os.path.join(save_dir, 'predictions.png')
        plt.savefig(output_path)
        print(f"Prediction visualization saved to {output_path}")
    
    plt.show()

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Make predictions with VGG model on any image')
    parser.add_argument('--images', nargs='+', type=str, required=True, 
                        help='Paths to input images (can specify multiple)')
    parser.add_argument('--model', type=str, default=None,
                        help='Path to saved model checkpoint (optional)')
    parser.add_argument('--output-dir', type=str, default='predictions',
                        help='Directory to save visualization results')
    args = parser.parse_args()
    
    # Load the model
    model, device = load_model(args.model)
    
    # Process each image and make predictions
    predictions = []
    
    for image_path in args.images:
        print(f"Processing {image_path}...")
        # Preprocess image
        image_tensor, original_image = preprocess_image(image_path, device)
        
        # Make prediction
        probs, indices = make_prediction(model, image_tensor)
        
        # Store results
        predictions.append((original_image, probs, indices))
        
        # Print text results
        if probs is not None:
            print(f"Predictions for {image_path}:")
            for i, (prob, idx) in enumerate(zip(probs, indices)):
                print(f"  {i+1}. {CIFAR10_CLASSES[idx]}: {prob*100:.2f}%")
            print()
    
    # Visualize all predictions
    visualize_predictions(args.images, predictions, args.output_dir)

if __name__ == "__main__":
    main()
