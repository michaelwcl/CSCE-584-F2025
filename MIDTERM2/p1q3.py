import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.optim.lr_scheduler import StepLR


class ResidualBlock(nn.Module):
    """
    Residual block with skip connection.
    Implements: out = F(x) + x
    """
    def __init__(self, in_channels, out_channels, stride=1, kernel_size=3):
        super(ResidualBlock, self).__init__()
        
        # Main path (F(x))
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=kernel_size,
            stride=stride, padding=kernel_size//2, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=kernel_size,
            stride=1, padding=kernel_size//2, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Skip connection (identity or projection)
        self.skip = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            # Projection shortcut for dimension mismatch
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
        self.relu2 = nn.ReLU(inplace=True)
    
    def forward(self, x):
        identity = self.skip(x)
        
        # Main path
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        # Skip connection: add identity
        out = out + identity
        out = self.relu2(out)
        
        return out


class ResNet(nn.Module):
    def __init__(self, num_classes=10, depth=56):
        super(ResNet, self).__init__()
        
        # Initial convolution layer
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        
        # Residual blocks: 3 groups of residual blocks
        # Group 1: 64 channels
        self.layer1 = self._make_layer(64, 64, num_blocks=depth//3, stride=1)
        
        # Group 2: 128 channels
        self.layer2 = self._make_layer(64, 128, num_blocks=depth//3, stride=2)
        
        # Group 3: 256 channels
        self.layer3 = self._make_layer(128, 256, num_blocks=depth//3, stride=2)
        
        # Global average pooling and classification head
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)
        
        # Initialize weights
        self._init_weights()
    
    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = []
        # First block with stride change if needed
        layers.append(ResidualBlock(in_channels, out_channels, stride=stride))
        # Remaining blocks with stride=1
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Initial convolution
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        # Residual blocks
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        # Classification head
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in train_loader:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        
        # Forward pass
        with torch.amp.autocast('cuda' if torch.cuda.is_available() else 'cpu'):
            outputs = model(images)
            loss = criterion(outputs, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Statistics
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    epoch_loss = total_loss / len(train_loader)
    epoch_acc = 100.0 * correct / total
    
    return epoch_loss, epoch_acc


def validate_epoch(model, val_loader, criterion, device):
    """Validate for one epoch."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            
            with torch.amp.autocast('cuda' if torch.cuda.is_available() else 'cpu'):
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    epoch_loss = total_loss / len(val_loader)
    epoch_acc = 100.0 * correct / total
    
    return epoch_loss, epoch_acc


def main():
    # Configuration
    BATCH_SIZE = 256  # Increased batch size for GPU efficiency
    NUM_EPOCHS = 50
    LEARNING_RATE = 0.1
    WEIGHT_DECAY = 1e-4
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # GPU optimization
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
        torch.cuda.empty_cache()
        torch.backends.cudnn.benchmark = True  # Optimize for fixed input sizes
        torch.backends.cudnn.deterministic = False  # Allow non-deterministic for speed
    
    print(f"Using device: {DEVICE}")
    print(f"Training ResNet for {NUM_EPOCHS} epochs with batch size {BATCH_SIZE}")
    
    # Data preprocessing
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2023, 0.1994, 0.2010]
        )
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2023, 0.1994, 0.2010]
        )
    ])
    
    # Load CIFAR-10 dataset
    print("Loading CIFAR-10 dataset...")
    train_dataset = datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train
    )
    val_dataset = datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_test
    )
    test_dataset = datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test
    )
    
    # Split training set into train and validation (90-10 split)
    train_size = int(0.9 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, _ = torch.utils.data.random_split(
        train_dataset, [train_size, val_size]
    )
    val_dataset, _ = torch.utils.data.random_split(
        val_dataset, [val_size, train_size]
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    # Create model
    model = ResNet(num_classes=10, depth=56).to(DEVICE)
    print(f"Model created: ResNet-56")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss().to(DEVICE)
    optimizer = optim.SGD(
        model.parameters(), lr=LEARNING_RATE, momentum=0.9,
        weight_decay=WEIGHT_DECAY, nesterov=True
    )
    scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
    
    # Training history
    train_losses = []
    train_accs = []
    val_accs = []
    
    # Training loop
    print("\nStarting training...")
    for epoch in range(NUM_EPOCHS):
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, DEVICE
        )
        val_loss, val_acc = validate_epoch(
            model, val_loader, criterion, DEVICE
        )
        
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        scheduler.step()
        
        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch [{epoch+1}/{NUM_EPOCHS}] - "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
                f"Val Acc: {val_acc:.2f}%"
            )
    
    # Test accuracy
    print("\nEvaluating on test set...")
    test_loss, test_acc = validate_epoch(
        model, test_loader, criterion, DEVICE
    )
    print(f"\nFinal Test Accuracy: {test_acc:.2f}%")
    
    # Plot training and validation accuracy
    plt.figure(figsize=(12, 5))
    
    # Accuracy plot
    plt.subplot(1, 2, 1)
    epochs_range = range(1, NUM_EPOCHS + 1)
    plt.plot(epochs_range, train_accs, 'b-', label='Training Accuracy')
    plt.plot(epochs_range, val_accs, 'r-', label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('ResNet-56 Training and Validation Accuracy')
    plt.legend()
    plt.grid(True)
    
    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, train_losses, 'g-', label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('ResNet-56 Training Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('resnet_training_results.png', dpi=150)
    print("Plot saved as 'resnet_training_results.png'")
    plt.show()
    
    print("\n" + "="*60)
    print("ARCHITECTURE SUMMARY")
    print("="*60)
    print(f"Model: ResNet-56 for CIFAR-10")
    print(f"Input: 3x32x32 images")
    print(f"Layer 1: 64 filters (18 residual blocks), stride=1")
    print(f"Layer 2: 128 filters (18 residual blocks), stride=2")
    print(f"Layer 3: 256 filters (18 residual blocks), stride=2")
    print(f"Global Average Pooling -> FC (256 -> 10)")
    print(f"\nEach residual block contains:")
    print(f"  - Conv2d(3x3) -> BatchNorm -> ReLU")
    print(f"  - Conv2d(3x3) -> BatchNorm")
    print(f"  - Skip connection (identity or projection)")
    print(f"  - ReLU activation")
    print(f"\nRegularization: Weight decay (L2) = {WEIGHT_DECAY}")
    print(f"Optimization: SGD with momentum=0.9, learning rate scheduling")
    print(f"Data augmentation: Random crop and horizontal flip")
    print("="*60)
    print(f"\nFINAL TEST ACCURACY: {test_acc:.2f}%")
    print("="*60)


if __name__ == "__main__":
    main()
