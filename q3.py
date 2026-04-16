import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.utils.data
import os
import time
from torch.utils.data.distributed import DistributedSampler

from config import NUM_WORKERS, LR, MOMENTUM, WEIGHT_DECAY
from dataloader import get_dataloader
from resnet18 import ResNet18
from train import train_epoch


def setup():
    dist.init_process_group("nccl")

def cleanup():
    dist.destroy_process_group()


def make_timed_hook(comm_times: list):
    """DDP comm hook that times each all-reduce bucket."""
    def hook(state, bucket):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fut = dist.all_reduce(bucket.buffer(), async_op=True).get_future()

        def done(f):
            torch.cuda.synchronize()
            comm_times.append(time.perf_counter() - t0)
            return f.value()[0]

        return fut.then(done)
    return hook


def test_batch_size(batch_size: int, local_rank: int):
    torch.cuda.empty_cache()
    device = torch.device("cuda", local_rank)

    try:
        base_loader = get_dataloader("./data", batch_size=batch_size,
                                     num_workers=NUM_WORKERS, train=True)
        sampler = DistributedSampler(base_loader.dataset)
        loader = torch.utils.data.DataLoader(
            base_loader.dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=NUM_WORKERS,
            pin_memory=True,
        )

        model = ResNet18(num_classes=10).to(device)
        comm_times = []
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
        model.register_comm_hook(state=None, hook=make_timed_hook(comm_times))

        optimizer = optim.SGD(model.parameters(), lr=LR,
                              momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
        criterion = nn.CrossEntropyLoss()

        # warm up
        comm_times.clear()
        train_epoch(model, loader, optimizer, criterion, device)

        # record
        comm_times.clear()
        loss, acc, dt, tt, tot = train_epoch(model, loader, optimizer, criterion, device)

        # train_time (tt) = CPU→GPU + forward + backward + optimizer step
        # comm (all-reduce) is embedded inside backward; hook measures it directly
        comm_time = sum(comm_times)
        compute_time = tt - comm_time

        return True, compute_time, comm_time, tot

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            torch.cuda.empty_cache()
            return False, 0.0, 0.0, 0.0
        raise


def bandwidth_utilization(model: nn.Module, comm_time_sec: float, world_size: int) -> float:
    """
    Ring all-reduce transfers 2*(N-1)/N * M bytes per GPU,
    where M = total gradient bytes, N = world_size.
    Bandwidth utilization (GB/s) = bytes_transferred / comm_time.
    """
    num_params = sum(p.numel() for p in model.parameters())
    M = num_params * 4  # float32 = 4 bytes
    N = world_size
    bytes_transferred = 2 * (N - 1) / N * M
    if comm_time_sec <= 0:
        return float('inf')
    return bytes_transferred / comm_time_sec / 1e9  # GB/s


if __name__ == "__main__":
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    setup()

    world_size = dist.get_world_size()

    if local_rank == 0:
        print(f"Computation vs Communication — {world_size} GPU(s)\n")
        print(f"{'batch':>8}  {'compute(s)':>12}  {'comm(s)':>10}  {'total(s)':>10}  {'BW(GB/s)':>10}")

    k, factor = 32, 4
    prev_k = k

    # Reference model for param count (bandwidth formula)
    ref_model = ResNet18(num_classes=10)

    while True:
        success, compute_time, comm_time, total_time = test_batch_size(k, local_rank)

        if not success:
            if local_rank == 0:
                print(f"\nOOM at batch size {k}")
                print(f"Max usable batch size: {prev_k}")
            break

        bw = bandwidth_utilization(ref_model, comm_time, world_size)
        if local_rank == 0:
            print(f"{k:>8}  {compute_time:>12.3f}  {comm_time:>10.3f}  {total_time:>10.3f}  {bw:>10.3f}")

        prev_k = k
        k *= factor

    if local_rank == 0:
        num_params = sum(p.numel() for p in ref_model.parameters())
        M = num_params * 4
        N = world_size
        print(f"\n--- Q3.2 Bandwidth Utilization ---")
        print(f"Model params : {num_params:,}  ({M/1e6:.2f} MB, float32)")
        print(f"Formula      : bytes_transferred = 2*(N-1)/N * M  =  2*{N-1}/{N} * {M/1e6:.2f} MB = {2*(N-1)/N*M/1e6:.2f} MB")
        print(f"Formula      : BW_utilization (GB/s) = bytes_transferred / comm_time")

    cleanup()
