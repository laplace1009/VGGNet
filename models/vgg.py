import torch
from torch import nn

class VGG(nn.Module):
    """VGG network implementation based on the paper 'Very Deep Convolutional Networks for Large-Scale Image Recognition'
    
    'Very Deep Convolutional Networks for Large-Scale Image Recognition' 논문을 기반으로 한 VGG 네트워크 구현
    """
    def __init__(self, config):
        """Initialize VGG network
        
        VGG 네트워크 초기화
        
        Args:
            config: Configuration object containing network parameters
                   네트워크 매개변수를 포함하는 구성 객체
        """
        super().__init__()
        self.config = config
        # 특징 추출 부분: convolution 계층 등
        self.features = self.make_layers()
        # AdaptiveAvgPool을 별도 속성으로 설정
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        # 분류 부분: fully connected 계층
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(),
            nn.Dropout(config.drop_p),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(config.drop_p),
            nn.Linear(4096, config.num_classes)
        )
        
        if config.init_weights:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """Forward pass through the network
        
        네트워크를 통한 순방향 전달
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
               (배치_크기, 채널, 높이, 너비) 형태의 입력 텐서
        
        Returns:
            Output tensor of shape (batch_size, num_classes)
            (배치_크기, 클래스_수) 형태의 출력 텐서
        """
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def make_layers(self):
        """Create feature extraction layers based on configuration
        
        구성에 기반한 특징 추출 레이어 생성
        
        Returns:
            nn.Sequential: Feature extraction layers
                          특징 추출 레이어
        """
        layers = []
        in_channels, config_layers, is_batch_norm = self.config.in_channels, self.config.layers, self.config.batch_norm
        for v in config_layers:
            if isinstance(v, int):
                layers.append(nn.Conv2d(in_channels, v, kernel_size=3, padding=1))
                if is_batch_norm:
                    layers.append(nn.BatchNorm2d(v))
                layers.append(nn.ReLU())
                in_channels = v
            else:
                layers.append(nn.MaxPool2d(2))
        return nn.Sequential(*layers)