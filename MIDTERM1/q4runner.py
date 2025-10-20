from __future__ import annotations

import time

#WARNING: use cuda
#pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
#source: https://pytorch.org/get-started/locally/

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from q4 import create_model, count_parameters, device

EPOCHS = 10


def train(model: nn.Module, loader, criterion, lr: float = 0.01):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(loader):
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


def run_training(model: nn.Module, train_loader, test_loader, lr: float = 0.01, epochs: int = 1):
    criterion = nn.CrossEntropyLoss().to(device)
    print(f'Using device: {device}')
    start_time = time.time()
    
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train(model, train_loader, criterion, lr=lr)
        t1 = time.time()
        print(f'Epoch {epoch}/{epochs} - Train loss: {train_loss:.4f}, Train acc: {train_acc*100:.2f}% (time: {t1-t0:.1f}s)')

    test_loss, test_acc = evaluate(model, test_loader, criterion)
    total_time = time.time() - start_time
    print(f'Test  - Loss: {test_loss:.4f}, Acc: {test_acc*100:.2f}%')
    print(f'Total time: {total_time:.1f}s')
    
    return test_loss, test_acc

def main():
    batch_size = 64
    test_batch_size = 1000
    lr = 0.01
    seed = 1

    torch.manual_seed(seed)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    train_dataset = datasets.MNIST('.', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('.', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)

    model = create_model()
    print(f'Model created. Trainable parameters: {count_parameters(model):,}')
    
    run_training(model, train_loader, test_loader, lr=lr, epochs=EPOCHS)


if __name__ == '__main__':
    main()
