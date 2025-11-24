"""
Graph Neural Network for Node Classification on Citeseer Dataset
This implementation uses PyTorch Geometric for GNN modeling on the Citeseer dataset.
"""

import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv, GATConv, SAGEConv
from torch_geometric.transforms import NormalizeFeatures
from torch.optim.lr_scheduler import StepLR
import numpy as np
import matplotlib.pyplot as plt



class CiteseerDatasetLoader:
    """Handles loading and preprocessing of Citeseer dataset."""
    
    def __init__(self, root='./data'):
        """
        Load Citeseer dataset from PyTorch Geometric.
        
        Args:
            root: Directory to store dataset
        """
        # Load Citeseer from Planetoid
        self.dataset = Planetoid(root=root, name='CiteSeer', transform=NormalizeFeatures())
        self.data = self.dataset[0]  # Single graph dataset
        
        # Set train/val/test masks if not already present
        if not hasattr(self.data, 'train_mask'):
            self._create_splits()
    
    def _create_splits(self):
        """Create train/val/test splits if not provided."""
        num_nodes = self.data.num_nodes
        indices = torch.randperm(num_nodes)
        
        # Standard split: 60% train, 20% val, 20% test
        train_size = int(0.6 * num_nodes)
        val_size = int(0.2 * num_nodes)
        
        train_idx = indices[:train_size]
        val_idx = indices[train_size:train_size + val_size]
        test_idx = indices[train_size + val_size:]
        
        self.data.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        self.data.val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        self.data.test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        
        self.data.train_mask[train_idx] = True
        self.data.val_mask[val_idx] = True
        self.data.test_mask[test_idx] = True
    
    def get_data(self):
        """Return the dataset."""
        return self.data
    
    def print_dataset_info(self):
        """Print detailed dataset information."""
        num_classes = int(self.data.y.max().item()) + 1
        print("\n" + "="*70)
        print("CITESEER DATASET INFORMATION")
        print("="*70)
        print(f"Number of nodes: {self.data.num_nodes}")
        print(f"Number of edges: {self.data.num_edges}")
        print(f"Number of node features: {self.data.num_node_features}")
        print(f"Number of classes: {num_classes}")
        print(f"\nFeature dimensions: {self.data.x.shape}")
        print(f"Edge index shape: {self.data.edge_index.shape}")
        print(f"Label shape: {self.data.y.shape}")
        print(f"\nTrain samples: {self.data.train_mask.sum().item()}")
        print(f"Val samples: {self.data.val_mask.sum().item()}")
        print(f"Test samples: {self.data.test_mask.sum().item()}")
        print(f"\nClass distribution (training set):")
        train_labels = self.data.y[self.data.train_mask]
        for c in range(num_classes):
            count = (train_labels == c).sum().item()
            print(f"  Class {c}: {count} samples")
        print("="*70 + "\n")

#GAT
class GraphAttentionNetwork(torch.nn.Module):
    
    def __init__(self, num_features, num_hidden=256, num_heads=8, 
                 num_classes=6, dropout=0.3):
        super(GraphAttentionNetwork, self).__init__()
        
        # Model parameters
        self.num_features = num_features
        self.num_hidden = num_hidden
        self.num_heads = num_heads
        self.num_classes = num_classes
        self.dropout_rate = dropout
        self.head_dim = num_hidden // num_heads
        
        # Attention layers
        self.gat1 = GATConv(num_features, self.head_dim, heads=num_heads, 
                            dropout=dropout, concat=True)
        self.gat2 = GATConv(num_hidden, self.head_dim, heads=num_heads, 
                            dropout=dropout, concat=True)
        self.gat3 = GATConv(num_hidden, num_classes, heads=1, 
                            dropout=dropout, concat=False)
        
        self.dropout = dropout
    
    def forward(self, x, edge_index):
        """
        Forward pass through the network.
        
        Args:
            x: Node feature matrix (N, num_features)
            edge_index: Edge index tensor (2, num_edges)
            
        Returns:
            logits: Output logits (N, num_classes)
        """
        # Layer 1: GAT -> ELU -> Dropout
        x = self.gat1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Layer 2: GAT -> ELU -> Dropout
        x = self.gat2(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Layer 3: GAT (output layer, no activation)
        x = self.gat3(x, edge_index)
        
        return x


class GraphConvolutionalNetwork(torch.nn.Module):
    def __init__(self, num_features, num_hidden_1=256, num_hidden_2=128,
                 num_hidden_3=64, num_classes=6, dropout=0.4):
        super(GraphConvolutionalNetwork, self).__init__()
        
        # Model parameters
        self.num_features = num_features
        self.num_hidden_1 = num_hidden_1
        self.num_hidden_2 = num_hidden_2
        self.num_hidden_3 = num_hidden_3
        self.num_classes = num_classes
        self.dropout_rate = dropout
        
        # Graph convolutional layers with batch normalization
        self.conv1 = GCNConv(num_features, num_hidden_1)
        self.bn1 = torch.nn.BatchNorm1d(num_hidden_1)
        
        self.conv2 = GCNConv(num_hidden_1, num_hidden_2)
        self.bn2 = torch.nn.BatchNorm1d(num_hidden_2)
        
        self.conv3 = GCNConv(num_hidden_2, num_hidden_3)
        self.bn3 = torch.nn.BatchNorm1d(num_hidden_3)
        
        self.conv4 = GCNConv(num_hidden_3, num_classes)
        
        self.dropout = dropout
    
    def forward(self, x, edge_index):
        # Layer 1: GCN -> BatchNorm -> ReLU -> Dropout
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Layer 2: GCN -> BatchNorm -> ReLU -> Dropout
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Layer 3: GCN -> BatchNorm -> ReLU -> Dropout
        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Layer 4: GCN (output layer, no activation)
        x = self.conv4(x, edge_index)
        
        return x


# ============================================================================
# TRAINING AND EVALUATION
# ============================================================================
class NodeClassificationTrainer:
    """Handles training and evaluation of node classification models."""
    
    def __init__(self, model, data, device='cpu', learning_rate=0.01, weight_decay=5e-4):
        self.model = model.to(device)
        self.data = data.to(device)
        self.device = device
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        
        # Optimizer: Adam with L2 regularization
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = StepLR(self.optimizer, step_size=50, gamma=0.5)
        
        # Tracking metrics
        self.train_losses = []
        self.val_accs = []
        self.test_accs = []
    
    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        self.optimizer.zero_grad()
        
        # Forward pass
        out = self.model(self.data.x, self.data.edge_index)
        
        # Compute loss on training set only
        loss = F.cross_entropy(out[self.data.train_mask], self.data.y[self.data.train_mask])
        
        # Backward pass
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    @torch.no_grad()
    def evaluate(self, mask):
        self.model.eval()
        out = self.model(self.data.x, self.data.edge_index)
        pred = out.argmax(dim=1)
        correct = (pred[mask] == self.data.y[mask]).sum().item()
        accuracy = correct / mask.sum().item()
        return accuracy
    
    def train(self, num_epochs=500, patience=100):
        """
        Train the model with early stopping.
        
        Args:
            num_epochs: Maximum number of epochs
            patience: Early stopping patience
        """
        print("\n" + "="*70)
        print("TRAINING GRAPH NEURAL NETWORK")
        print("="*70)
        print(f"Device: {self.device}")
        print(f"Learning rate: {self.learning_rate}")
        print(f"Weight decay (L2): {self.weight_decay}")
        print(f"Dropout rate: {self.model.dropout_rate}")
        print("="*70 + "\n")
        
        best_val_acc = 0
        best_test_acc = 0
        patience_counter = 0
        best_model_state = None
        
        for epoch in range(1, num_epochs + 1):
            train_loss = self.train_epoch()
            val_acc = self.evaluate(self.data.val_mask)
            test_acc = self.evaluate(self.data.test_mask)
            
            self.train_losses.append(train_loss)
            self.val_accs.append(val_acc)
            self.test_accs.append(test_acc)
            
            # Early stopping check based on validation accuracy
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_test_acc = test_acc
                patience_counter = 0
                best_model_state = self.model.state_dict().copy()
            else:
                patience_counter += 1
            
            if epoch % 50 == 0 or epoch == 1:
                print(f"Epoch {epoch:3d} | Loss: {train_loss:.4f} | "
                      f"Val Acc: {val_acc:.4f} | Test Acc: {test_acc:.4f}")
            
            self.scheduler.step()
            
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch}")
                # Restore best model
                self.model.load_state_dict(best_model_state)
                break
        
        print("="*70 + "\n")
    
    def evaluate_final(self):
        """Evaluate final model performance."""
        train_acc = self.evaluate(self.data.train_mask)
        val_acc = self.evaluate(self.data.val_mask)
        test_acc = self.evaluate(self.data.test_mask)
        
        return train_acc, val_acc, test_acc
    
    def plot_results(self, save_path='p4.png'):
        """Plot training results."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Loss plot
        axes[0].plot(self.train_losses, label='Training Loss', linewidth=2)
        axes[0].set_xlabel('Epoch', fontsize=12)
        axes[0].set_ylabel('Loss', fontsize=12)
        axes[0].set_title('Training Loss Over Epochs', fontsize=14, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend(fontsize=10)
        
        # Accuracy plot
        axes[1].plot(self.val_accs, label='Validation Accuracy', linewidth=2)
        axes[1].plot(self.test_accs, label='Test Accuracy', linewidth=2)
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('Accuracy', fontsize=12)
        axes[1].set_title('Validation and Test Accuracy Over Epochs', fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        axes[1].legend(fontsize=10)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training results saved to {save_path}")
        plt.close()


def main():
    """Main execution function."""
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Determine device - prefer CUDA
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"\nGPU Available: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
    else:
        device = torch.device('cpu')
        print("\nWarning: CUDA not available, falling back to CPU")
    
    print(f"Using device: {device}")
    
    print("\n" + "="*70)
    print("LOADING CITESEER DATASET")
    print("="*70)
    loader = CiteseerDatasetLoader(root='./data')
    data = loader.get_data()
    num_classes = int(data.y.max().item()) + 1
    loader.print_dataset_info()

    print("="*70)
    print("GNN MODEL ARCHITECTURE - GRAPH ATTENTION NETWORK (GAT)")
    print("="*70)
    model = GraphAttentionNetwork(
        num_features=data.num_node_features,
        num_hidden=256,
        num_heads=8,
        num_classes=num_classes,
        dropout=0.3
    )
    
    print(model)
    print("\nModel Configuration:")
    print(f"  Input dimension: {data.num_node_features}")
    print(f"  Hidden dimension: 256 (8 heads × 32)")
    print(f"  Attention heads (hidden): 8")
    print(f"  Attention heads (output): 1")
    print(f"  Output dimension: {num_classes}")
    print(f"  Dropout rate: 0.3")
    print(f"  Activation: ELU")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters())}")
    print("="*70)
    
    trainer = NodeClassificationTrainer(
        model=model,
        data=data,
        device=device,
        learning_rate=0.01,
        weight_decay=5e-4
    )
    
    trainer.train(num_epochs=500, patience=100)
    print("\n" + "="*70)
    print("FINAL RESULTS")
    print("="*70)
    train_acc, val_acc, test_acc = trainer.evaluate_final()
    
    print(f"Training Accuracy:   {train_acc:.4f} ({train_acc*100:.2f}%)")
    print(f"Validation Accuracy: {val_acc:.4f} ({val_acc*100:.2f}%)")
    print(f"Test Accuracy:       {test_acc:.4f} ({test_acc*100:.2f}%)")
    print("="*70 + "\n")
    
    # Plot results
    trainer.plot_results()
    
    return model, trainer, data


if __name__ == '__main__':
    model, trainer, data = main()
