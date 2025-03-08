import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from models.vgg import VGG
from utils.configs import Config
import time
import numpy as np
from tqdm import tqdm

def parse_args():
    """Parse command line arguments for testing
    
    테스트를 위한 명령줄 인수 파싱
    
    Returns:
        argparse.Namespace: Parsed arguments
                           파싱된 인수들
    """
    parser = argparse.ArgumentParser(description='Test VGG Network')
    parser.add_argument('--data-dir', type=str, default='./data', help='Dataset directory')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for testing')
    parser.add_argument('--vgg-config', type=str, default='D', choices=['A', 'B', 'D', 'E'], help='VGG configuration (A, B, D, E)')
    parser.add_argument('--batch-norm', action='store_true', help='Use batch normalization')
    parser.add_argument('--num-classes', type=int, default=10, help='Number of classes in dataset')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA evaluation')
    parser.add_argument('--seed', type=int, default=1, help='Random seed')
    parser.add_argument('--checkpoint-path', type=str, required=True, help='Path to the model checkpoint')
    parser.add_argument('--visualize', action='store_true', help='Visualize model predictions')
    return parser.parse_args()

def create_test_loader(args):
    """Create data loader for testing
    
    테스트용 데이터 로더 생성
    
    Args:
        args: Command line arguments
             명령줄 인수
    
    Returns:
        DataLoader: Test data loader
                   테스트 데이터 로더
    """
    # Data transformations for testing
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # If using CIFAR-10 (modify as needed for other datasets)
    try:
        test_dataset = datasets.CIFAR10(root=args.data_dir, train=False, download=True, transform=test_transform)
    except Exception as e:
        print(f"Error loading CIFAR-10 dataset: {e}")
        print("Falling back to ImageNet (you need to manually download and place it in the data directory)...")
        # ImageFolder requires organized directories by class
        test_dataset = datasets.ImageFolder(os.path.join(args.data_dir, 'val'), transform=test_transform)
    
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=4, pin_memory=True
    )
    
    return test_loader, test_dataset.classes if hasattr(test_dataset, 'classes') else None

def evaluate_model(model, test_loader, criterion, device):
    """Evaluate model performance on test dataset
    
    테스트 데이터셋에서 모델 성능 평가
    
    Args:
        model: The neural network model to evaluate
              평가할 신경망 모델
        test_loader: DataLoader with test data
                    테스트 데이터가 담긴 DataLoader
        criterion: Loss function
                  손실 함수
        device: Device to evaluate on (cuda/cpu)
               평가에 사용할 장치 (cuda/cpu)
                   
    Returns:
        tuple: (test_loss, test_accuracy) - Overall test loss and accuracy
               (테스트_손실, 테스트_정확도) - 전체 테스트 손실과 정확도
    """
    """Evaluate model performance on the test dataset"""
    model.eval()
    test_loss = 0
    correct = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in tqdm(test_loader, desc="Evaluating"):
            data, target = data.to(device), target.to(device)
            output = model(data)
            
            # Sum up batch loss
            test_loss += criterion(output, target).item()
            
            # Get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\n')
    
    return accuracy, all_preds, all_targets

def visualize_results(model, test_loader, class_names, device, num_images=5):
    """Visualize model predictions on sample images
    
    샘플 이미지에 대한 모델 예측 결과 시각화
    
    Args:
        model: The neural network model to use for predictions
              예측에 사용할 신경망 모델
        test_loader: DataLoader with test data
                    테스트 데이터가 담긴 DataLoader
        class_names: List of class names
                    클래스 이름 목록
        device: Device to run inference on (cuda/cpu)
               추론에 사용할 장치 (cuda/cpu)
        num_images: Number of images to visualize
                   시각화할 이미지 수
    """
    """Visualize model predictions on sample images"""
    try:
        import matplotlib.pyplot as plt
        from matplotlib import gridspec
        
        # Get a batch of test images
        dataiter = iter(test_loader)
        images, labels = next(dataiter)
        images, labels = images[:num_images], labels[:num_images]
        
        # Move images to device
        images = images.to(device)
        
        # Get predictions
        with torch.no_grad():
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
        
        # Move data back to CPU for visualization
        images = images.cpu().numpy()
        labels = labels.cpu().numpy()
        preds = preds.cpu().numpy()
        
        # Create figure with subplots
        fig = plt.figure(figsize=(15, 5))
        gs = gridspec.GridSpec(1, num_images)
        
        # Plot images with predictions
        for i in range(num_images):
            ax = plt.subplot(gs[i])
            
            # Transpose image from [C, H, W] to [H, W, C] and denormalize
            img = images[i].transpose((1, 2, 0))
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img = std * img + mean
            img = np.clip(img, 0, 1)
            
            ax.imshow(img)
            ax.set_title(f'Pred: {class_names[preds[i]]}\nTrue: {class_names[labels[i]]}', 
                        color=('green' if preds[i] == labels[i] else 'red'))
            ax.axis('off')
        
        plt.tight_layout()
        plt.savefig('test_predictions.png')
        print("Saved visualization to test_predictions.png")
        
    except ImportError as e:
        print(f"Visualization requires matplotlib: {e}")
    except Exception as e:
        print(f"Error during visualization: {e}")

def main():
    """Main function to run the test process
    
    테스트 과정을 실행하는 메인 함수
    """
    args = parse_args()
    
    # Set the random seed for reproducibility
    torch.manual_seed(args.seed)
    
    # CUDA setup
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    if use_cuda:
        torch.cuda.manual_seed(args.seed)
    
    # Create test data loader
    test_loader, class_names = create_test_loader(args)
    
    # Initialize the VGG configuration
    # 이전 방식: config = Config(config=args.vgg_config, batch_norm=args.batch_norm, num_classes=args.num_classes)
    # 새로운 방식: 팩토리 메서드 사용 또는 vgg_type 매개변수 사용
    config = Config(vgg_type=args.vgg_config, batch_norm=args.batch_norm, num_classes=args.num_classes)
    config.make_layers()
    
    # Create the model
    model = VGG(config).to(device)
    print(f"Created VGG-{args.vgg_config}{'_BN' if args.batch_norm else ''} model")
    
    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint_path}")
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    
    # Handle both state_dict only and full checkpoint dictionary
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')} with accuracy {checkpoint.get('accuracy', 'unknown')}")
    else:
        model.load_state_dict(checkpoint)
        print("Loaded model state_dict")
    
    # Loss function for evaluation
    criterion = nn.CrossEntropyLoss()
    
    # Evaluate model
    start_time = time.time()
    accuracy, all_preds, all_targets = evaluate_model(model, test_loader, criterion, device)
    end_time = time.time()
    
    print(f"Evaluation completed in {end_time - start_time:.2f} seconds")
    
    # Visualize results if requested
    if args.visualize and class_names:
        visualize_results(model, test_loader, class_names, device)

if __name__ == '__main__':
    main()
