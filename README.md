# VGGNet Implementation

## 프로젝트 개요

VGGNet은 2014년 옥스포드 대학의 Visual Geometry Group(VGG)에서 개발한 심층 합성곱 신경망(CNN) 아키텍처입니다. 이 프로젝트는 PyTorch를 사용하여 VGGNet의 다양한 버전을 구현하고, 이를 학습 및 테스트하는 코드를 제공합니다.

VGGNet은 단순하지만 효과적인 3x3 합성곱 필터를 사용하여 깊은 네트워크 구조를 통해 이미지 분류 성능을 개선했습니다. 본 구현은 VGG-11, VGG-13, VGG-16, VGG-19를 포함한 모든 VGG 변형 모델을 지원합니다.

## 프로젝트 스펙

### 모델 아키텍처

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

### 기술 스택

- **프레임워크**: PyTorch
- **언어**: Python 3.6+
- **주요 라이브러리**:
  - torchvision (데이터 로딩 및 변환)
  - numpy (수치 연산)
  - tqdm (진행 상황 표시)
  - matplotlib (시각화, 테스트용)

### 주요 기능

- 다양한 VGG 구성 (A, B, D, E) 지원
- 배치 정규화 옵션
- CIFAR-10 데이터셋 자동 다운로드 및 사용
- ImageNet 형식 데이터셋 지원
- 학습 중 체크포인트 저장
- 학습률 스케줄링
- 테스트 결과 시각화

## 사용 방법

### 요구 사항 설치

```bash
pip install torch torchvision numpy tqdm matplotlib
```

### 모델 학습

기본 CIFAR-10 데이터셋으로 VGG-16 모델(구성 D)을 학습하려면:

```bash
python train.py --vgg-config D --batch-size 64 --epochs 50
```

배치 정규화를 적용한 VGG-19 모델(구성 E)을 학습하려면:

```bash
python train.py --vgg-config E --batch-norm --batch-size 32 --epochs 100
```

### 학습 매개변수

주요 학습 매개변수:

- `--data-dir`: 데이터셋 디렉토리 (기본값: './data')
- `--batch-size`: 학습 배치 크기 (기본값: 32)
- `--epochs`: 학습 에폭 수 (기본값: 50)
- `--lr`: 학습률 (기본값: 0.001)
- `--vgg-config`: VGG 구성 선택 [A, B, D, E] (기본값: 'D')
- `--batch-norm`: 배치 정규화 사용 여부 (플래그)
- `--num-classes`: 분류할 클래스 수 (기본값: 10)
- `--save-model`: 학습된 모델 저장 여부 (기본값: True)

### 모델 테스트

학습된 모델을 테스트하려면:

```bash
python test.py --checkpoint-path ./checkpoints/vgg_D_bn_True_best.pth --vgg-config D --batch-norm
```

결과 시각화를 포함한 테스트:

```bash
python test.py --checkpoint-path ./checkpoints/vgg_E_bn_True_best.pth --vgg-config E --batch-norm --visualize
```

### 테스트 매개변수

주요 테스트 매개변수:

- `--checkpoint-path`: 학습된 모델 체크포인트 경로 (필수)
- `--vgg-config`: VGG 구성 선택 [A, B, D, E] (기본값: 'D')
- `--batch-norm`: 배치 정규화 사용 여부 (플래그)
- `--visualize`: 테스트 결과 시각화 여부 (플래그)

## 프로젝트 구조

```
├── data/                  # 데이터셋 저장 디렉토리
├── models/                # 모델 구현 디렉토리
│   └── vgg.py            # VGG 모델 구현
├── utils/                 # 유틸리티 스크립트
│   └── configs.py        # VGG 구성 설정
├── checkpoints/           # 모델 체크포인트 저장 디렉토리
├── train.py               # 학습 스크립트
└── test.py                # 테스트 스크립트
```

## 참고 자료

- [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556) - K. Simonyan and A. Zisserman (2014)