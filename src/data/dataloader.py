import random

import numpy
import torch
import torchvision
from torch.utils.data import DataLoader

from src.config.config import Config


def seed_worker(worker_id):
    """Seed dataloader workers."""
    worker_seed = torch.initial_seed() % 2**32
    numpy.random.seed(worker_seed)
    random.seed(worker_seed)


def get_dataloader(config: Config) -> tuple[DataLoader, DataLoader]:
    """Creates dataloader for specified dataset."""

    dataset = config.dataloader.dataset
    num_workers = config.dataloader.num_workers
    batch_size = config.trainer.batch_size

    if dataset == "cifar10":

        cifar10 = torchvision.datasets.CIFAR10(root="./data", train=True, download=True)

        # mean = [0.49139968 0.48215841 0.44653091]
        mean = numpy.mean(numpy.array(cifar10.data / 255.0), axis=(0, 1, 2))

        # std = [0.24703223 0.24348513 0.26158784]
        std = numpy.std(numpy.array(cifar10.data / 255.0), axis=(0, 1, 2))

        transform_train = torchvision.transforms.Compose(
            [
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.RandomVerticalFlip(),
                torchvision.transforms.ColorJitter(brightness=0.5, hue=0.3),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.RandomErasing(),
                torchvision.transforms.Normalize(mean, std),
            ]
        )

        transform_test = torchvision.transforms.Compose(
            [torchvision.transforms.ToTensor(), torchvision.transforms.Normalize(mean, std)]
        )

        train_dataset = torchvision.datasets.CIFAR10(
            root="./data", train=True, download=True, transform=transform_train
        )

        test_dataset = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform_test)

        # Add number of classes and input shape to config
        config.data.num_classes = 10
        config.data.input_shape = (3, 32, 32)

    else:
        raise Exception(f"Dataset {dataset} not available.")

    generator = torch.Generator()
    generator.manual_seed(config.random_seed)

    if "cuda" in str(config.trainer.device):
        pin_memory = True
    else:
        pin_memory = False

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        worker_init_fn=seed_worker,
        generator=generator,
        shuffle=True,
        pin_memory=pin_memory,
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=2 * batch_size,
        num_workers=num_workers,
        worker_init_fn=seed_worker,
        generator=generator,
        shuffle=True,
        pin_memory=pin_memory,
    )

    return train_loader, test_loader
