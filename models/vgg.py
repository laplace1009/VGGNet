import torch
from torch import nn

class VGG(nn.Module):
    def __init__(self, config):
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
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def make_layers(self):
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