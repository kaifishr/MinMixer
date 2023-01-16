"""Collection of custom layers for neural networks."""
import torch
from torch import nn
from torch.nn import functional as F


class SimpleComplexLinear(nn.Module):
    """Linear layer to simulation evolution from simple to complex."""
    def __init__(self, in_features: int, out_features: int, bias: bool=True):
        super().__init__()

        self.weight = torch.nn.Parameter(torch.empty((out_features, in_features)))
        self.bias = torch.nn.Parameter(torch.empty(out_features)) if bias else None

        self.rand = torch.nn.Parameter(torch.rand_like(input=self.weight), requires_grad=False)
        self.gate = torch.nn.Parameter(torch.zeros_like(self.weight), requires_grad=False)
        self.prob = torch.nn.Parameter(torch.tensor(0.0), requires_grad=False)

    @property
    def probability(self) -> float:
        return self.prob.item()

    @probability.setter
    def probability(self, prob: float) -> None:
        self.prob.copy_(max(0.0, min(prob, 1.0)))

    @torch.no_grad()
    def evolve(self):
        self.gate.data = torch.where(self.rand < self.prob, 1.0, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.gate * self.weight, self.bias)
        # if self._probability.item() < 1.0:
        #     return F.linear(x, self.gate * self.weight, self.bias)
        # else:
        #     return F.linear(x, self.weight, self.bias)


class Conv2d(nn.Conv2d):
    """Layer to simulation evolution from simple to complex."""

    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int, padding: str
    ):

        super(Conv2d, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
        )

        self.prob = None 
        self.rand =torch.nn.Parameter(torch.rand_like(input=self.weight), requires_grad=False)
        self.gate =torch.nn.Parameter(torch.zeros_like(self.weight), requires_grad=False)

    @torch.no_grad()
    def mask(self):
        self.gate.data = torch.where(self.rand < self.prob, 1.0, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(
            input=x,
            weight=self.gate * self.weight,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
        )
