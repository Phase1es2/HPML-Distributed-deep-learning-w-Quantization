import torch
import argparse

from config import NUM_WORKERS, LR, MOMENTUM, WEIGHT_DECAY
from dataloader import get_dataloader
from resnet18 import ResNet18
import torch.optim as optim
import torch.nn as nn

import torch.distributed as dist
import os
from torch.utils.data.distributed import DistributedSampler


from train import train_epoch


def setup():
    dist.init_process_group("nccl")

def cleanup():
    dist.destroy_process_group()

def run(batch_size: int, local_rank: int):
    torch.cuda.empty_cache()
    device = torch.device("cuda", local_rank)

    dataset = get_dataloader("./data", batch_size=batch_size,
                             num_workers=NUM_WORKERS, train=True)
    sampler = DistributedSampler(dataset.dataset)
    loader = torch.utils.data.DataLoader(
        dataset.dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=NUM_WORKERS,
    )
    model = ResNet18(num_classes=10).to(device)
    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[local_rank]
    )
    optimizer = optim.SGD(model.parameters(), lr=LR,
                          momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()

    # warm up (epoch 0, not counted)
    train_epoch(model, loader, optimizer, criterion, device)

    # train epochs 1–5, report only epoch 5
    for i in range(1, 6):
        loss, acc, dt, tt, tot = train_epoch(model, loader, optimizer, criterion, device)
        if local_rank == 0:
            print(f"  epoch {i}:  loss={loss:.4f}  acc={acc:.2f}%")

    if local_rank == 0:
        print(f"\nEpoch 5 result — loss: {loss:.4f}  acc: {acc:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, required=True,
                        help="Largest batch size per GPU found in Q1")
    args = parser.parse_args()

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    setup()

    if local_rank == 0:
        print(f"Large Batch Training — batch size {args.batch_size} per GPU, "
              f"{dist.get_world_size()} GPUs\n")

    run(args.batch_size, local_rank)

    cleanup()
