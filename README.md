# VGGNet Implementation

## 프로젝트 개요 (Project Overview)

VGGNet은 2014년 옥스포드 대학의 Visual Geometry Group(VGG)에서 개발한 심층 합성곱 신경망(CNN) 아키텍처입니다. 이 프로젝트는 PyTorch를 사용하여 VGGNet의 다양한 버전을 구현하고, 이를 학습 및 테스트하는 코드를 제공합니다.

VGGNet은 단순하지만 효과적인 3x3 합성곱 필터를 사용하여 깊은 네트워크 구조를 통해 이미지 분류 성능을 개선했습니다. 본 구현은 VGG-11, VGG-13, VGG-16, VGG-19를 포함한 모든 VGG 변형 모델을 지원합니다.

**English:**

VGGNet is a deep convolutional neural network (CNN) architecture developed by the Visual Geometry Group (VGG) at Oxford University in 2014. This project provides code to implement various versions of VGGNet using PyTorch, along with training and testing capabilities.

VGGNet improved image classification performance through a deep network structure using simple but effective 3x3 convolutional filters. This implementation supports all VGG variant models, including VGG-11, VGG-13, VGG-16, and VGG-19.

## 프로젝트 스펙 (Project Specifications)

### 모델 아키텍처 (Model Architecture)

- **VGG 구성**:
  - **VGG-11** (구성 A): 11개의 가중치 계층
  - **VGG-13** (구성 B): 13개의 가중치 계층
  - **VGG-16** (구성 D): 16개의 가중치 계층
  - **VGG-19** (구성 E): 19개의 가중치 계층
  
- **구현 특징**:
  - 3x3 컨볼루션 필터 사용
  - 맥스 풀링을 통한 다운샘플링
  - 옵션으로 배치 정규화 지원
  - 드롭아웃을 통한 과적합 방지
  - 완전 연결 계층을 통한 분류
  - 적응형 평균 풀링 계층 포함

### 기술 스택 (Technology Stack)

- **프레임워크**: PyTorch
- **언어**: Python 3.6+
- **주요 라이브러리**:
  - torchvision (데이터 로딩 및 변환)
  - numpy (수치 연산)
  - tqdm (진행 상황 표시)
  - matplotlib (시각화, 테스트용)

### 주요 기능 (Key Features)

- 다양한 VGG 구성 (A, B, D, E) 지원
- 배치 정규화 옵션
- CIFAR-10 데이터셋 자동 다운로드 및 사용
- ImageNet 형식 데이터셋 지원
- 학습 중 체크포인트 저장
- 학습률 스케줄링
- 테스트 결과 시각화

## 사용 방법 (Usage)

### 요구 사항 설치 (Installing Requirements)

```bash
pip install torch torchvision numpy tqdm matplotlib
```

### 모델 학습 (Model Training)

기본 CIFAR-10 데이터셋으로 VGG-16 모델(구성 D)을 학습하려면:

```bash
python train.py --vgg-config D --batch-size 64 --epochs 50
```

배치 정규화를 적용한 VGG-19 모델(구성 E)을 학습하려면:

```bash
python train.py --vgg-config E --batch-norm --batch-size 32 --epochs 100
```

### 학습 매개변수 (Training Parameters)

주요 학습 매개변수:

- `--data-dir`: 데이터셋 디렉토리 (기본값: './data')
- `--batch-size`: 학습 배치 크기 (기본값: 32)
- `--epochs`: 학습 에폭 수 (기본값: 50)
- `--lr`: 학습률 (기본값: 0.001)
- `--vgg-config`: VGG 구성 선택 [A, B, D, E] (기본값: 'D')
- `--batch-norm`: 배치 정규화 사용 여부 (플래그)
- `--num-classes`: 분류할 클래스 수 (기본값: 10)
- `--save-model`: 학습된 모델 저장 여부 (기본값: True)

### 모델 테스트 (Model Testing)

학습된 모델을 테스트하려면:

```bash
python test.py --checkpoint-path ./checkpoints/vgg_D_bn_True_best.pth --vgg-config D --batch-norm
```

결과 시각화를 포함한 테스트:

```bash
python test.py --checkpoint-path ./checkpoints/vgg_E_bn_True_best.pth --vgg-config E --batch-norm --visualize
```

### 테스트 매개변수 (Testing Parameters)

주요 테스트 매개변수:

- `--checkpoint-path`: 학습된 모델 체크포인트 경로 (필수)
- `--vgg-config`: VGG 구성 선택 [A, B, D, E] (기본값: 'D')
- `--batch-norm`: 배치 정규화 사용 여부 (플래그)
- `--visualize`: 테스트 결과 시각화 여부 (플래그)

## 프로젝트 구조 (Project Structure)

```
├── data/                  # 데이터셋 저장 디렉토리 (Dataset directory)
├── models/                # 모델 구현 디렉토리 (Model implementation directory)
│   └── vgg.py            # VGG 모델 구현 (VGG model implementation)
├── utils/                 # 유틸리티 스크립트 (Utility scripts)
│   └── configs.py        # VGG 구성 설정 (VGG configuration settings)
├── checkpoints/           # 모델 체크포인트 저장 디렉토리 (Model checkpoint directory)
├── train_cifar10.py       # CIFAR-10 데이터셋 학습 스크립트 (CIFAR-10 training script)
├── predict_any_image.py   # 이미지 예측 스크립트 (Image prediction script)
└── extract_sample_image.py # 샘플 이미지 추출 스크립트 (Sample image extraction script)
```

### 테스트 스크립트 (Test Scripts)

프로젝트에는 VGG 모델을 테스트하기 위한 세 가지 주요 스크립트가 포함되어 있습니다:

#### 1. CIFAR-10 학습 (train_cifar10.py)

CIFAR-10 데이터셋을 사용하여 VGG 모델을 학습하는 스크립트입니다.

```bash
# 기본 설정으로 CIFAR-10 학습 실행
python train_cifar10.py

# 더 많은 에폭과 다른 학습률로 학습 실행
python train_cifar10.py --epochs 10 --batch-size 128 --lr 0.01 --output my_model.pth
```

#### 2. 이미지 예측 (predict_any_image.py)

학습된 모델을 사용하여 임의의 이미지를 분류하는 스크립트입니다.

```bash
# 특정 이미지 예측
python predict_any_image.py --images path/to/image.jpg --model checkpoints/vgg_cifar10.pth

# 여러 이미지 예측
python predict_any_image.py --images image1.jpg image2.jpg image3.jpg --output-dir my_predictions
```

#### 3. 샘플 이미지 추출 (extract_sample_image.py)

CIFAR-10 데이터셋에서 테스트용 샘플 이미지를 추출하는 스크립트입니다.

```bash
# 샘플 이미지 추출
python extract_sample_image.py
```

**English:**

The project includes three main scripts for testing the VGG model:

#### 1. CIFAR-10 Training (train_cifar10.py)

Script to train the VGG model using the CIFAR-10 dataset.

```bash
# Run CIFAR-10 training with default settings
python train_cifar10.py

# Run training with more epochs and different learning rate
python train_cifar10.py --epochs 10 --batch-size 128 --lr 0.01 --output my_model.pth
```

#### 2. Image Prediction (predict_any_image.py)

Script to classify arbitrary images using the trained model.

```bash
# Predict a specific image
python predict_any_image.py --images path/to/image.jpg --model checkpoints/vgg_cifar10.pth

# Predict multiple images
python predict_any_image.py --images image1.jpg image2.jpg image3.jpg --output-dir my_predictions
```

#### 3. Sample Image Extraction (extract_sample_image.py)

Script to extract sample images from the CIFAR-10 dataset for testing.

```bash
# Extract sample images
python extract_sample_image.py
```

### 데이터셋 (Dataset)

이 프로젝트는 기본적으로 CIFAR-10 데이터셋을 사용합니다. CIFAR-10은 10개 클래스의 60,000개 컬러 이미지(32x32 픽셀)로 구성된 데이터셋입니다. 데이터셋은 처음 실행 시 자동으로 다운로드됩니다.

**English:**

This project uses the CIFAR-10 dataset by default. CIFAR-10 consists of 60,000 color images (32x32 pixels) in 10 classes. The dataset is automatically downloaded during the first run.

## 참고 자료 (References)

- [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556) - K. Simonyan and A. Zisserman (2014)
- [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html) - Alex Krizhevsky