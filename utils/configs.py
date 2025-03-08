from dataclasses import dataclass, field
from enum import Enum
from typing import List, Union

class VGGType(Enum):
    """VGG architecture variants with standard naming convention
    
    VGG 아키텍처 변형들의 표준 명명 규칙
    """
    VGG11 = "A"  # Also known as VGG11
    VGG13 = "B"  # Also known as VGG13
    VGG16 = "D"  # Also known as VGG16
    VGG19 = "E"  # Also known as VGG19

    @classmethod
    def from_string(cls, value: str) -> "VGGType":
        """Convert string input to VGGType, supporting both new and legacy naming
        
        문자열 입력을 VGGType으로 변환, 새로운 방식과 레거시 명명 방식 모두 지원
        """
        if value.upper().startswith("VGG"):
            # Handle new style names like "VGG16"
            try:
                return cls[value.upper()]
            except KeyError:
                pass
        
        # Handle legacy style names (A, B, D, E)
        for vgg_type in cls:
            if vgg_type.value == value:
                return vgg_type
        
        raise ValueError(f"Invalid VGG type: {value}. Use one of {[t.name for t in cls]} "
                         f"or legacy types {[t.value for t in cls]}")

@dataclass
class Config:
    """Configuration for VGG architectures
    
    VGG 아키텍처를 위한 설정 클래스
    
    Args:
        vgg_type: VGG architecture variant (VGG11, VGG13, VGG16, VGG19 or legacy A, B, D, E)
               VGG 아키텍처 변형 (VGG11, VGG13, VGG16, VGG19 또는 레거시 A, B, D, E)
        in_channels: Number of input channels (3 for RGB images)
                   입력 채널 수 (RGB 이미지의 경우 3)
        batch_norm: Whether to use batch normalization
                   배치 정규화 사용 여부
        num_classes: Number of output classes
                   출력 클래스 수
        init_weights: Whether to initialize weights
                    가중치 초기화 여부
        drop_p: Dropout probability
               드롭아웃 확률
        kernel_size: Convolutional kernel size
                   컨볼루션 커널 크기
        padding: Padding size for convolutions
                컨볼루션 패딩 크기
    """
    vgg_type: Union[str, VGGType]
    layers: List[Union[int, str]] = field(default_factory=list)
    in_channels: int = 3
    batch_norm: bool = False
    num_classes: int = 1000
    init_weights: bool = True
    drop_p: float = 0.5
    kernel_size: int = 3
    padding: int = 1
    
    def __post_init__(self):
        """Convert string vgg_type to Enum if needed and validate parameters
        
        필요한 경우 문자열 vgg_type을 Enum으로 변환하고 매개변수 유효성 검사
        """
        if isinstance(self.vgg_type, str):
            self.vgg_type = VGGType.from_string(self.vgg_type)
        self._validate_parameters()
    
    def _validate_parameters(self):
        """Validate configuration parameters
        
        구성 매개변수 유효성 검사
        """
        if not isinstance(self.vgg_type, VGGType):
            raise TypeError(f"vgg_type must be VGGType or string, got {type(self.vgg_type)}")
        if self.in_channels <= 0:
            raise ValueError(f"in_channels must be positive, got {self.in_channels}")
        if not 0 <= self.drop_p <= 1:
            raise ValueError(f"drop_p must be between 0 and 1, got {self.drop_p}")
        if self.num_classes <= 0:
            raise ValueError(f"num_classes must be positive, got {self.num_classes}")
    
    def make_layers(self):
        """Set up the network architecture based on the VGG variant
        
        VGG 변형에 따라 네트워크 아키텍처 설정
        """
        # Use the legacy config value (A, B, D, E) for backward compatibility
        config_value = self.vgg_type.value
        
        if config_value == "A":  # VGG11
            self.layers = [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"]
        elif config_value == "B":  # VGG13
            self.layers = [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"]
        elif config_value == "D":  # VGG16
            self.layers = [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"]
        elif config_value == "E":  # VGG19
            self.layers = [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"]
        return self
    
    @classmethod
    def vgg11(cls, **kwargs) -> "Config":
        """Create VGG11 configuration
        
        VGG11 구성 생성
        """
        return cls(vgg_type=VGGType.VGG11, **kwargs).make_layers()
    
    @classmethod
    def vgg13(cls, **kwargs) -> "Config":
        """Create VGG13 configuration
        
        VGG13 구성 생성
        """
        return cls(vgg_type=VGGType.VGG13, **kwargs).make_layers()
    
    @classmethod
    def vgg16(cls, **kwargs) -> "Config":
        """Create VGG16 configuration
        
        VGG16 구성 생성
        """
        return cls(vgg_type=VGGType.VGG16, **kwargs).make_layers()
    
    @classmethod
    def vgg19(cls, **kwargs) -> "Config":
        """Create VGG19 configuration
        
        VGG19 구성 생성
        """
        return cls(vgg_type=VGGType.VGG19, **kwargs).make_layers()
