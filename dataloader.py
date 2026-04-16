import torch
import torchvision
import torchvision.transforms as transforms
from config import CIFAR10_MEAN, CIFAR10_STD


def get_dataloader(data_path: str, batch_size: int,
                   num_workers: int, train: bool = True) -> torch.utils.data.DataLoader:
    if train:
        transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
        ])

    dataset = torchvision.datasets.CIFAR10(
        root=data_path, train=train, download=True, transform=transform,
    )
    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size,
        num_workers=num_workers, pin_memory=True,
        multiprocessing_context='spawn' if num_workers > 0 else None,
    )
