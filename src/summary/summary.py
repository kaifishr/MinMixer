"""Holds methods for Tensorboard.
"""
import math

import matplotlib.pyplot as plt
import numpy
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader


def add_graph(model: nn.Module, dataloader: DataLoader, writer: SummaryWriter, config: dict) -> None:
    """Add graph of model to Tensorboard.

    Args:
        model:
        dataloader:
        writer:
        config:

    """
    device = config.trainer.device
    x_data, _ = next(iter(dataloader))
    writer.add_graph(model=model, input_to_model=x_data.to(device))


def add_linear_weights(
    writer: SummaryWriter,
    model: nn.Module,
    global_step: int,
    n_samples_max: int = 64,
    do_rescale: bool = False,
) -> None:
    """Adds visualization of channel and token embeddings to Tensorboard."""
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            weight = module.weight.detach().cpu().numpy()

            height, width = weight.shape
            dim = int(math.sqrt(width))

            if not dim**2 == width:
                continue

            # Extract samples
            n_samples = min(height, n_samples_max)
            weight = weight[:n_samples]

            # Rescale
            if do_rescale:
                x_min = numpy.min(weight, axis=-1, keepdims=True)
                x_max = numpy.max(weight, axis=-1, keepdims=True)
                weight = (weight - x_min) / (x_max - x_min + 1e-6)

            # Reshape
            weight = weight.reshape(-1, dim, dim)

            # Plot weights
            ncols = 8
            nrows = int(math.ceil(n_samples / ncols))
            figsize = (0.5 * ncols, 0.5 * nrows)

            figure, axes = plt.subplots(
                nrows=nrows,
                ncols=ncols,
                figsize=figsize,
                layout="constrained",
            )

            for ax, w in zip(axes.flatten(), weight):
                ax.imshow(w, cmap="bwr", interpolation="none")  # none, spline16, ...

            for ax in axes.flatten():
                ax.axis("off")

            writer.add_figure(name, figure, global_step)
