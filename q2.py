import torch

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

def test_batch_size(batch_size: int, local_rank: int):
    torch.cuda.empty_cache()
    device = torch.device("cuda", local_rank)

    try:
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

        # warm up
        train_epoch(model, loader, optimizer, criterion, device)

        # record
        loss, acc, dt, tt, tot = train_epoch(model, loader, optimizer, criterion, device)
        if local_rank == 0:
            print(f"{loss:>8.4f}  {acc:>7.2f}  {dt:>10.3f}  {tt:>10.3f}  {tot:>10.3f}")

        return True

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            torch.cuda.empty_cache()
            return False
        else:
            raise e


if __name__ == "__main__":
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    setup()

    if local_rank == 0:
        print("Speedup Measurement\n")
        print(f"{'loss':>8}  {'acc(%)':>7}  {'data(s)':>10}  {'train(s)':>10}  {'total(s)':>10}")

    k, factor = 32, 4
    prev_k = k
    num_gpu = dist.get_world_size()

    while True:
        #if local_rank == 0:
        #     print(f"Training with batch size {k}")
        success = test_batch_size(k, local_rank)

        if not success:
            if local_rank == 0:
                print(f"\nOOM at batch size {k}")
                print(f"Max usable batch size: {prev_k}")
            break
        else:
            prev_k = k
            k *= factor

    cleanup()
