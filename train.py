import torch
import torch.nn as nn
import torch.optim as optim
import time

def sync(device: torch.device):
    if device.type == 'cuda':
        torch.cuda.synchronize()

def train_epoch(model: nn.Module, loader: torch.utils.data.DataLoader,
                optimizer: optim.Optimizer, criterion: nn.Module,
                device: torch.device):
    model.train()
    total_loss = 0.0
    correct = 0.0
    total = 0
    data_time_sum = 0.0

    epoch_start  = time.perf_counter()
    sync(device)
    t_data_start = time.perf_counter()

    for inputs, labels in loader:
        sync(device)
        data_time_sum += time.perf_counter() - t_data_start

        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

        sync(device)
        t_data_start = time.perf_counter()

    sync(device)
    epoch_end = time.perf_counter()

    avg_loss = total_loss / total
    accuracy = 100 * correct / total
    total_time = epoch_end - epoch_start
    train_time = total_time - data_time_sum
    return avg_loss, accuracy, data_time_sum, train_time, total_time



