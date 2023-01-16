"""Collection of custom neural networks."""
import torch
import torch.nn as nn

from src.config import Config
from src.modules.module import MixerBlock
from src.modules.module import ImageToSequence 
from src.modules.module import Classifier
from src.modules.layer import SimpleComplexLinear


class MLPMixer(nn.Module):
    """MLPMixer neural network."""

    def __init__(self, config: Config):
        """Initializes MLPMixer."""
        super().__init__()

        self.image_to_sequence = ImageToSequence(config)

        num_blocks = config.model.num_blocks
        mixer_blocks = [MixerBlock(config) for _ in range(num_blocks)]
        self.mixer_blocks = nn.Sequential(*mixer_blocks)

        self.classifier = Classifier(config=config)

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        """Initializes weights for all modules."""
        if isinstance(module, (nn.Linear, SimpleComplexLinear)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.01)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.image_to_sequence(x)
        x = self.mixer_blocks(x)
        x = self.classifier(x)
        return x
