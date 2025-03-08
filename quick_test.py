import torch
import torch.nn as nn
from models.vgg import VGG
from utils.configs import Config, VGGType
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

def main(num_epochs=1):
    print("==== VGG 모델 테스트 ====")
    
    # 가장 작은 VGG 모델인 VGG11 사용 (최소한의 계산량)
    print("1. VGG11 모델 생성 중...")
    config = Config.vgg11(batch_norm=True, num_classes=10)  # 팩토리 메서드 사용
    model = VGG(config)
    
    # 모델 아키텍처 출력
    print(f"\n모델 구조:")
    print(f"- 특성 추출 레이어: {len(model.features)} 레이어")
    print(f"- 분류기 레이어: {len(model.classifier)} 레이어")
    
    # 모델 파라미터 수 계산
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"- 총 파라미터: {total_params:,}")
    print(f"- 학습 가능한 파라미터: {trainable_params:,}")
    
    # 더 자세한 모델 정보 출력 (선택 사항)
    print("\n2. 상세 모델 구조:")
    print(model)
    
    # 작은 데이터셋 로드 (CIFAR-10의 일부)
    print("\n3. 테스트용 미니 데이터셋 로드 중...")
    transform = transforms.Compose([
        transforms.Resize(224),  # VGG는 224x224 입력 기대
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # CIFAR-10 데이터셋의 일부만 사용 (더 빠른 테스트를 위해)
    dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    subset_size = 1000  # 1000개 샘플만 사용
    subset_indices = torch.randperm(len(dataset))[:subset_size]
    subset = torch.utils.data.Subset(dataset, subset_indices)
    dataloader = DataLoader(subset, batch_size=32, shuffle=True)
    
    print(f"- 데이터셋 크기: {len(subset)} 샘플")
    print(f"- 배치 크기: 32")
    print(f"- 배치 수: {len(dataloader)}")
    
    # 모델 GPU로 이동 (가능한 경우)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"\n4. 사용 중인 디바이스: {device}")
    model = model.to(device)
    
    # 모델 입출력 테스트
    print("\n5. 단일 배치로 순전파 테스트 중...")
    sample_batch, sample_labels = next(iter(dataloader))
    sample_batch = sample_batch.to(device)
    
    # 입력 형태 출력
    print(f"- 입력 텐서 크기: {sample_batch.shape}")
    
    # 순전파 테스트
    with torch.no_grad():
        output = model(sample_batch)
    
    # 출력 형태 출력
    print(f"- 출력 텐서 크기: {output.shape}")
    print(f"- 예상 출력 크기: [배치 크기, 클래스 수] = [{sample_batch.shape[0]}, {config.num_classes}]")
    
    # 예측 클래스 확인
    _, predicted = torch.max(output, 1)
    print(f"- 첫 5개 샘플의 예측 클래스: {predicted[:5].cpu().numpy()}")
    
    # 샘플 이미지 시각화
    print("\n6. 샘플 이미지 시각화 중...")
    sample_img = sample_batch[0].cpu().permute(1, 2, 0).numpy()
    # 정규화 해제
    mean = torch.tensor([0.485, 0.456, 0.406]).reshape(1, 1, 3).numpy()
    std = torch.tensor([0.229, 0.224, 0.225]).reshape(1, 1, 3).numpy()
    sample_img = std * sample_img + mean
    sample_img = sample_img.clip(0, 1)
    
    plt.figure(figsize=(6, 6))
    plt.imshow(sample_img)
    plt.title(f"예측: {predicted[0].item()}")
    plt.axis('off')
    plt.savefig('sample_prediction.png')
    print("- 샘플 이미지와 예측이 'sample_prediction.png'에 저장되었습니다.")
    
    print(f"\n7. 간단한 학습 테스트 ({num_epochs} 에폭)...")
    # 손실 함수와 옵티마이저 설정
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    
    # num_epochs 에폭 학습
    model.train()
    epoch_losses = []
    epoch_accuracies = []
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        batch_count = 0
        
        print(f"\n  에폭 {epoch+1}/{num_epochs} 학습 중...")
        for inputs, labels in dataloader:
            batch_count += 1
            inputs, labels = inputs.to(device), labels.to(device)
            
            # 그래디언트 초기화
            optimizer.zero_grad()
            
            # 순전파
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # 역전파 및 최적화
            loss.backward()
            optimizer.step()
            
            # 통계
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            if batch_count % 5 == 0:
                print(f"    배치 {batch_count}/{len(dataloader)}, 손실: {running_loss/batch_count:.4f}, 정확도: {100*correct/total:.2f}%")
    
        # 에폭 종료 후 통계 저장
        epoch_loss = running_loss/len(dataloader)
        epoch_accuracy = 100*correct/total
        epoch_losses.append(epoch_loss)
        epoch_accuracies.append(epoch_accuracy)
        print(f"  에폭 {epoch+1} 완료: 손실 = {epoch_loss:.4f}, 정확도 = {epoch_accuracy:.2f}%")
    
    print(f"\n테스트 완료!")
    print(f"- 첫 에폭 손실: {epoch_losses[0]:.4f}")
    print(f"- 마지막 에폭 손실: {epoch_losses[-1]:.4f}")
    print(f"- 첫 에폭 정확도: {epoch_accuracies[0]:.2f}%")
    print(f"- 마지막 에폭 정확도: {epoch_accuracies[-1]:.2f}%")
    
    if num_epochs > 1:
        # 여러 에폭에 걸친 손실 및 정확도 변화 시각화
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(range(1, num_epochs+1), epoch_losses)
        plt.title('에폭별 손실')
        plt.xlabel('에폭')
        plt.ylabel('손실')
        
        plt.subplot(1, 2, 2)
        plt.plot(range(1, num_epochs+1), epoch_accuracies)
        plt.title('에폭별 정확도')
        plt.xlabel('에폭')
        plt.ylabel('정확도 (%)')
        
        plt.tight_layout()
        plt.savefig('training_progress.png')
        print("- 학습 진행 상황이 'training_progress.png'에 저장되었습니다.")
    
    print(f"- 이것은 간단한 테스트이므로 정확도가 낮을 수 있습니다.")
    
    print("\n전체 모델 테스트가 성공적으로 완료되었습니다!")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='VGG 모델 테스트')
    parser.add_argument('--epochs', type=int, default=1, help='학습할 에폭 수 (기본값: 1)')
    args = parser.parse_args()
    
    main(num_epochs=args.epochs)
