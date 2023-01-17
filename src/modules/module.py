"""Common modules for neural networks."""
import torch
import torch.nn as nn

from src.config import Config


class ImageToSequence(nn.Module):
    """Image to sequence module.

    Transforms image into sequence using embeddings.

    Images are of size (batch_size, num_channels, width, height) and
    are transformed to size (batch_size, sequence_size, embedding_dim).

    Attributes:
        sequence_length:
        embedding_dim:
    """

    def __init__(self, config: Config) -> None:
        """Initializes ImageToSequence module."""
        super().__init__()

        self.sequence_length = config.model.input_sequence_length
        self.embedding_dim = config.model.embedding_dim

        patch_size = config.model.image_to_sequence.patch_size
        img_channels, img_height, img_width = config.data.input_shape

        assert (img_height % patch_size == 0) and (img_width % patch_size == 0)

        self.conv = nn.Conv2d(
            in_channels=img_channels,
            out_channels=self.sequence_length,
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size),
        )

        self.linear = nn.Linear(
            in_features=(img_height // patch_size) * (img_width // patch_size),
            out_features=self.embedding_dim,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = torch.flatten(input=x, start_dim=2, end_dim=-1)
        x = self.linear(x)
        x = x.view(-1, self.sequence_length, self.embedding_dim)
        return x


class MlpBlock(nn.Module):
    """Classic MLP block

    MLP block as used in the original MLP-Mixer paper.

    """

    def __init__(self, dim: int, config: Config) -> None:
        super().__init__()

        expansion_factor = config.model.expansion_factor

        hidden_dim = int(expansion_factor * dim)

        self.mlp_block = nn.Sequential(
            nn.Linear(in_features=dim, out_features=hidden_dim),
            nn.GELU(),
            nn.Linear(in_features=hidden_dim, out_features=dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp_block(x)


class SwapAxes(nn.Module):
    def __init__(self, axis0: int, axis1):
        super().__init__()
        self.axis0 = axis0
        self.axis1 = axis1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.swapaxes(x, axis0=self.axis0, axis1=self.axis1)


class MixerBlock(nn.Module):
    """MLP Mixer block

    Mixes channel and token dimension one after the other.
    """

    def __init__(self, config: Config):
        super().__init__()

        sequence_length = config.model.input_sequence_length
        embedding_dim = config.model.embedding_dim

        self.token_mixer = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            SwapAxes(axis0=-2, axis1=-1),
            MlpBlock(dim=sequence_length, config=config),
            SwapAxes(axis0=-2, axis1=-1),
        )

        self.channel_mixer = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            MlpBlock(dim=embedding_dim, config=config),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.token_mixer(x)
        x = x + self.channel_mixer(x)
        return x


class Classifier(nn.Module):
    """Classifier head."""

    def __init__(self, config: Config) -> None:
        """Initializes the classifier."""
        super().__init__()
        sequence_length = config.model.input_sequence_length
        embedding_dim = config.model.embedding_dim
        num_classes = config.data.num_classes

        self.linear = nn.Linear(
            in_features=sequence_length * embedding_dim,
            out_features=num_classes,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method."""
        x = torch.flatten(x, start_dim=1)
        out = self.linear(x)
        return out
