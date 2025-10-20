#WARNING!!!!
#this code uses a massive amount of cpu usage. i have 7950x3d and training took like an hour
#therefore ive added gpu/cuda support. ensure you have a compatible gpu and also cuda installed:

#pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
#source: https://pytorch.org/get-started/locally/

#this will decrease the training time significantly
#tested on a RTX 3090, this program used up to 14GB of VRAM

from __future__ import annotations

import time

EPOCHS = 60
#results seem to plateu around 60  epochs, so adjust as needed


import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from q5 import create_model, device


def train_one_epoch(model: nn.Module, loader, criterion, lr: float = 0.01):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for _, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)
        for param in model.parameters():
            if param.grad is not None:
                param.grad.zero_()
        
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        
        with torch.no_grad():
            for param in model.parameters():
                if param.grad is not None:
                    param.add_(param.grad, alpha=-lr)

        running_loss += loss.item() * data.size(0)
        preds = output.argmax(dim=1)
        correct += (preds == target).sum().item()
        total += data.size(0)

    avg_loss = running_loss / total
    acc = correct / total
    return avg_loss, acc


def evaluate(model: nn.Module, loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            running_loss += loss.item() * data.size(0)
            preds = output.argmax(dim=1)
            correct += (preds == target).sum().item()
            total += data.size(0)

    avg_loss = running_loss / total
    acc = correct / total
    return avg_loss, acc


def main():
    batch_size = 128
    test_batch_size = 1000
    lr = 0.01
    seed = 1

    torch.manual_seed(seed)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])

    train_dataset = datasets.CIFAR10('.', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10('.', train=False, download=True, transform=transform)

    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)

    model = create_model()
    #print(f'parameters: {count_parameters(model):,}') uncomment for debugging

    criterion = nn.CrossEntropyLoss().to(device)
    print(f'using device: {device}')
    
    start_time = time.time()
    for epoch in range(1, EPOCHS + 1):
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, lr)
        t1 = time.time()
        print(f'Epoch {epoch} - Train loss: {train_loss:.4f}, Train acc: {train_acc*100:.2f}% (time: {t1-t0:.1f}s)')

    test_loss, test_acc = evaluate(model, test_loader, criterion)
    total_time = time.time() - start_time
    print(f'Test  - Loss: {test_loss:.4f}, Acc: {test_acc*100:.2f}%')
    print(f'Total time: {total_time:.1f}s')


if __name__ == '__main__':
    main()
