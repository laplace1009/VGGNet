from dataclasses import dataclass, field

@dataclass
class Config:
    config: str
    layers: list = field(default_factory=list)
    in_channels: int = 3
    batch_norm: bool = False
    num_classes: int = 1000
    init_weights: bool = True
    drop_p: float = 0.5

    def make_layers(self):
        if self.config == "A":
            self.layers = [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"]
        elif self.config == "B":
            self.layers = [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"]
        elif self.config == "D":
            self.layers = [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"]
        elif self.config == "E":
            self.layers = [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"]
        else:
            raise ValueError("Invalid config")
        return self
