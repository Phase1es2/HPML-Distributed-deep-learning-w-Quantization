import torch

from config import NUM_WORKERS, LR, MOMENTUM, WEIGHT_DECAY
from dataloader import get_dataloader
from resnet18 import ResNet18
import torch.optim as optim
import torch.nn as nn

from train import train_epoch


def test_batch_size(batch_size: int):
    torch.cuda.empty_cache()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        loader = get_dataloader("./data", batch_size=batch_size,
                                num_workers=NUM_WORKERS, train=True)
        model = ResNet18(num_classes=10).to(device)
        optimizer = optim.SGD(model.parameters(), lr=LR,
                              momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
        criterion = nn.CrossEntropyLoss()

        # warm up
        train_epoch(model, loader, optimizer, criterion, device)

        # record
        loss, acc, dt, tt, tot = train_epoch(model, loader, optimizer, criterion, device)
        print(f"{loss:>8.4f}  {acc:>7.2f}  {dt:>10.3f}  {tt:>10.3f}  {tot:>10.3f}")

        return True

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            torch.cuda.empty_cache()
            return False
        else:
            raise e


if __name__ == "__main__":
    print("Computational Efficiency w.r.t Batch Size\n")
    print(f"{'loss':>8}  {'acc(%)':>7}  {'data(s)':>10}  {'train(s)':>10}  {'total(s)':>10}")

    k, factor = 32, 4
    prev_k = k

    while True:
        print(f"Training with batch size {k}")
        success = test_batch_size(k)

        if not success:
            print(f"\nOOM at batch size {k}")
            print(f"Max usable batch size: {prev_k}")
            break
        else:
            prev_k = k
            k *= factor