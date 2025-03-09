import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
import argparse

from models.vgg import VGG
from utils.configs import Config

def train_model(num_epochs=5, batch_size=64, learning_rate=0.001, save_path='vgg_cifar10.pth'):
    # Check for CUDA
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create the output directory if it doesn't exist
    os.makedirs('checkpoints', exist_ok=True)
    checkpoint_path = os.path.join('checkpoints', save_path)
    
    # Data transformation
    transform_train = transforms.Compose([
        transforms.Resize(224),  # VGG expects 224x224 input
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    transform_test = transforms.Compose([
        transforms.Resize(224),  # VGG expects 224x224 input
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load CIFAR-10 dataset
    print("Loading CIFAR-10 dataset...")
    trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    
    # For faster training, limit the dataset size (optional)
    subset_size = 10000  # Use 10,000 samples for training
    test_subset_size = 2000  # Use 2,000 samples for testing
    
    # Create subset indices
    train_indices = torch.randperm(len(trainset))[:subset_size]
    test_indices = torch.randperm(len(testset))[:test_subset_size]
    
    # Create subsets
    train_subset = torch.utils.data.Subset(trainset, train_indices)
    test_subset = torch.utils.data.Subset(testset, test_indices)
    
    # Create data loaders
    trainloader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=2)
    testloader = DataLoader(test_subset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    print(f"Training dataset size: {len(train_subset)}")
    print(f"Testing dataset size: {len(test_subset)}")
    
    # Initialize model
    print("Creating VGG11 model...")
    config = Config.vgg11(batch_norm=True, num_classes=10)
    model = VGG(config)
    model = model.to(device)
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    # Track training progress
    train_losses = []
    train_accuracies = []
    test_accuracies = []
    
    # Train model
    print(f"Starting training for {num_epochs} epochs...")
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for i, (inputs, labels) in enumerate(trainloader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            # Track statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Print progress
            if (i+1) % 20 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, Batch {i+1}/{len(trainloader)}, Loss: {running_loss/(i+1):.4f}, Acc: {100*correct/total:.2f}%")
        
        # Calculate epoch statistics
        epoch_loss = running_loss / len(trainloader)
        epoch_acc = 100 * correct / total
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc)
        
        # Evaluate on test set
        model.eval()
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for inputs, labels in testloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()
        
        test_acc = 100 * test_correct / test_total
        test_accuracies.append(test_acc)
        
        print(f"Epoch {epoch+1} completed: Loss={epoch_loss:.4f}, Train Acc={epoch_acc:.2f}%, Test Acc={test_acc:.2f}%")
        
        # Update learning rate
        scheduler.step()
    
    print("Training completed!")
    
    # Save the model
    torch.save({
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': epoch_loss,
        'train_acc': epoch_acc,
        'test_acc': test_acc
    }, checkpoint_path)
    print(f"Model saved to {checkpoint_path}")
    
    # Plot training progress
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(train_losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    plt.subplot(1, 3, 2)
    plt.plot(train_accuracies)
    plt.title('Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    
    plt.subplot(1, 3, 3)
    plt.plot(test_accuracies)
    plt.title('Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    print("Training history plotted and saved to 'training_history.png'")
    
    return checkpoint_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train VGG on CIFAR-10')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs (default: 5)')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size (default: 64)')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate (default: 0.001)')
    parser.add_argument('--output', type=str, default='vgg_cifar10.pth', help='Output model filename (default: vgg_cifar10.pth)')
    args = parser.parse_args()
    
    trained_model_path = train_model(
        num_epochs=args.epochs,
        batch_size=args.batch_size, 
        learning_rate=args.lr,
        save_path=args.output
    )
