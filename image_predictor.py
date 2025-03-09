import os
import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import argparse
from models.vgg import VGG
from utils.configs import Config

# CIFAR-10 class names
CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

def load_model(model_path=None):
    """Load a VGG model, either from a saved checkpoint or create a new one"""
    # Create a VGG11 model with batch normalization
    config = Config.vgg11(batch_norm=True, num_classes=10)
    model = VGG(config)
    
    # If a model path is provided, load the saved weights
    if model_path and os.path.exists(model_path):
        print(f"Loading model from {model_path}")
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        print("Using a new VGG11 model (not trained)")
    
    # Move model to the appropriate device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()  # Set to evaluation mode
    
    return model, device

def process_image(image_path, device):
    """Process an image to be suitable for the VGG model"""
    # VGG preprocessing: resize to 224x224 and normalize with ImageNet stats
    transform = transforms.Compose([
        transforms.Resize(224),
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

def predict(model, image_tensor):
    """Make a prediction with the model"""
    with torch.no_grad():
        outputs = model(image_tensor)
        
    # Get probabilities
    probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
    
    # Get top predictions
    top_probs, top_indices = torch.topk(probabilities, 3)
    
    return top_probs.cpu().numpy(), top_indices.cpu().numpy()

def display_prediction(image, top_probs, top_indices, save_path=None):
    """Display the image with prediction results"""
    # Create figure with two panels
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Display image
    ax1.imshow(image)
    ax1.axis('off')
    ax1.set_title("Input Image")
    
    # Display predictions as a bar chart
    y_pos = range(len(top_indices))
    class_names = [CIFAR10_CLASSES[i] for i in top_indices]
    ax2.barh(y_pos, top_probs * 100, align='center')
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(class_names)
    ax2.set_xlabel('Probability (%)')
    ax2.set_title('Top Predictions')
    
    # Add text with exact probabilities
    for i, (prob, idx) in enumerate(zip(top_probs, top_indices)):
        ax2.text(prob * 100 + 1, i, f"{prob*100:.1f}%", va='center')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Prediction saved to {save_path}")
    
    plt.show()

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Test VGG Model with Images')
    parser.add_argument('--image', type=str, required=True, 
                        help='Path to the input image')
    parser.add_argument('--model', type=str, default=None,
                        help='Path to saved model checkpoint (optional)')
    parser.add_argument('--output', type=str, default='prediction_result.png',
                        help='Path to save the prediction visualization')
    args = parser.parse_args()
    
    # Load the model
    model, device = load_model(args.model)
    
    # Process the image
    image_tensor, original_image = process_image(args.image, device)
    
    # Make prediction
    top_probs, top_indices = predict(model, image_tensor)
    
    # Display results
    display_prediction(original_image, top_probs, top_indices, args.output)
    
    # Print text results
    print("\nTop Predictions:")
    for i, (prob, idx) in enumerate(zip(top_probs, top_indices)):
        print(f"{i+1}. {CIFAR10_CLASSES[idx]}: {prob*100:.2f}%")

if __name__ == "__main__":
    main()
