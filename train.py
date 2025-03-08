import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from models.vgg import VGG
from utils.configs import Config
import time
import copy

def parse_args():
    """Parse command line arguments for training
    
    훈련을 위한 명령줄 인수 파싱
    
    Returns:
        argparse.Namespace: Parsed arguments
                           파싱된 인수들
    """
    parser = argparse.ArgumentParser(description='Train VGG Network')
    parser.add_argument('--data-dir', type=str, default='./data', help='Dataset directory')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--test-batch-size', type=int, default=32, help='Batch size for testing')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
    parser.add_argument('--weight-decay', type=float, default=5e-4, help='Weight decay')
    parser.add_argument('--vgg-config', type=str, default='D', choices=['A', 'B', 'D', 'E'], help='VGG configuration (A, B, D, E)')
    parser.add_argument('--batch-norm', action='store_true', help='Use batch normalization')
    parser.add_argument('--num-classes', type=int, default=10, help='Number of classes in dataset')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, help='Random seed')
    parser.add_argument('--log-interval', type=int, default=10, help='How many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=True, help='Save the model')
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints', help='Directory to save checkpoints')
    return parser.parse_args()

def create_data_loaders(args):
    """Create data loaders for training and validation
    
    훈련 및 검증을 위한 데이터 로더 생성
    
    Args:
        args: Command line arguments
             명령줄 인수
    
    Returns:
        tuple: (train_loader, test_loader) - DataLoader objects for training and testing
               (훈련_로더, 테스트_로더) - 훈련 및 테스트용 DataLoader 객체
    """
    # Data transformations
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # If using CIFAR-10 (modify as needed for other datasets)
    try:
        train_dataset = datasets.CIFAR10(root=args.data_dir, train=True, download=True, transform=train_transform)
        test_dataset = datasets.CIFAR10(root=args.data_dir, train=False, download=True, transform=test_transform)
    except Exception as e:
        print(f"Error loading CIFAR-10 dataset: {e}")
        print("Falling back to ImageNet (you need to manually download and place it in the data directory)...")
        # ImageFolder requires organized directories by class
        train_dataset = datasets.ImageFolder(os.path.join(args.data_dir, 'train'), transform=train_transform)
        test_dataset = datasets.ImageFolder(os.path.join(args.data_dir, 'val'), transform=test_transform)
    
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=4, pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=args.test_batch_size, shuffle=False,
        num_workers=4, pin_memory=True
    )
    
    return train_loader, test_loader

def train_model(model, dataloaders, criterion, optimizer, scheduler, args, device):
    """Train the model and evaluate on validation set
    
    모델을 훈련하고 검증 세트에서 평가
    
    Args:
        model: The neural network model to train
              훈련할 신경망 모델
        dataloaders: Dictionary with 'train' and 'val' dataloaders
                    'train'과 'val' 데이터로더가 있는 사전
        criterion: Loss function
                  손실 함수
        optimizer: Optimization algorithm
                  최적화 알고리즘
        scheduler: Learning rate scheduler
                  학습률 스케줄러
        args: Command line arguments
             명령줄 인수
        device: Device to train on (cuda/cpu)
               훈련에 사용할 장치 (cuda/cpu)
    
    Returns:
        nn.Module: Best model based on validation accuracy
                  검증 정확도 기반 최고 성능 모델
    """
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    for epoch in range(args.epochs):
        print(f'Epoch {epoch}/{args.epochs - 1}')
        print('-' * 10)
        
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
                dataloader = dataloaders['train']
            else:
                model.eval()   # Set model to evaluate mode
                dataloader = dataloaders['val']
            
            running_loss = 0.0
            running_corrects = 0
            
            # Iterate over data
            for batch_idx, (inputs, labels) in enumerate(dataloader):
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                # Zero the parameter gradients
                optimizer.zero_grad()
                
                # Forward pass
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    
                    # Backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                
                if phase == 'train' and batch_idx % args.log_interval == 0:
                    print(f'Batch: {batch_idx}/{len(dataloader)}, Loss: {loss.item():.4f}')
            
            if phase == 'train' and scheduler is not None:
                scheduler.step()
                
            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_corrects.double() / len(dataloader.dataset)
            
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
            # Deep copy the model if it's the best validation performance
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                
                # Save the best model checkpoint
                if args.save_model:
                    if not os.path.exists(args.checkpoint_dir):
                        os.makedirs(args.checkpoint_dir)
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'accuracy': best_acc,
                    }, os.path.join(args.checkpoint_dir, f'vgg_{args.vgg_config}_bn_{args.batch_norm}_best.pth'))
        
        print()
    
    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best validation Acc: {best_acc:.4f}')
    
    # Load best model weights
    model.load_state_dict(best_model_wts)
    return model

def main():
    """Main function to run the training process
    
    훈련 과정을 실행하는 메인 함수
    """
    args = parse_args()
    
    # Set the random seed for reproducibility
    torch.manual_seed(args.seed)
    
    # CUDA setup
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    if use_cuda:
        torch.cuda.manual_seed(args.seed)
    
    # Create data loaders
    train_loader, test_loader = create_data_loaders(args)
    
    # Create dataloaders dictionary
    dataloaders = {
        'train': train_loader,
        'val': test_loader
    }
    
    # Initialize the VGG configuration
    # 이전 방식: config = Config(config=args.vgg_config, batch_norm=args.batch_norm, num_classes=args.num_classes)
    # 새로운 방식: 팩토리 메서드 사용 또는 vgg_type 매개변수 사용
    config = Config(vgg_type=args.vgg_config, batch_norm=args.batch_norm, num_classes=args.num_classes)
    config.make_layers()
    
    # Create the model
    model = VGG(config).to(device)
    print(f"Created VGG-{args.vgg_config}{'_BN' if args.batch_norm else ''} model")
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    
    # Train the model
    model = train_model(model, dataloaders, criterion, optimizer, scheduler, args, device)
    
    # Save the final model
    if args.save_model:
        if not os.path.exists(args.checkpoint_dir):
            os.makedirs(args.checkpoint_dir)
        torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, f'vgg_{args.vgg_config}_bn_{args.batch_norm}_final.pth'))

if __name__ == '__main__':
    main()
