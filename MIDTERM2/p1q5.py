"""
Graph Neural Network for Link Prediction on Cora Dataset

DATASET DESCRIPTION:
==================
The Cora dataset is a citation network where:
- Nodes: 2,708 scientific papers
- Edges: 5,429 citation relationships (undirected)
- Node Features: 1,433-dimensional bag-of-words feature vectors
- Node Labels: 7 classes (ML research categories)
- Each node has binary features indicating the presence/absence of specific words

TASK: Link Prediction
Predict whether an edge exists between two nodes in the graph.

NODE/EDGE FEATURE SETUP:
=======================
- Node Features: 1,433-dimensional bag-of-words vectors (raw node embeddings)
- Edge Features: None (implicit - derived from node embeddings)
- Edge Construction: Edges are derived from citation relationships
- Training/Test Split: 
  * 70% of edges for training
  * 30% of edges for testing (negative sampling included)

MODEL ARCHITECTURE:
===================
- Type: Graph Convolutional Network (GCN) with link prediction head
- Layers:
  * Input: 1,433 node features
  * GCN Layer 1: 1,433 -> 512 hidden units + ReLU activation
  * GCN Layer 2: 512 -> 256 hidden units + ReLU activation
  * GCN Layer 3: 256 -> 128 hidden units + ReLU activation
  * Link Prediction Head: Concatenates embeddings of two nodes and passes through MLP
    - MLP: 256 -> 64 -> 1 (binary classification with sigmoid)

- Dropout: 0.5 (applied after each GCN layer)
- Optimizer: Adam with learning rate 0.01
- Loss Function: Binary Cross Entropy (BCE)
- Batch Size: Full batch training
- Epochs: 100
"""

import torch
import torch.nn.functional as F
from torch.nn import Linear, Dropout
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.nn import GCNConv
from torch_geometric.utils import negative_sampling, train_test_split_edges
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


class GCNLinkPredictor(torch.nn.Module):
    
    def __init__(self, in_channels, hidden_channels, out_channels, dropout_rate=0.5):
        super().__init__()
        
        # GCN Encoder Layers
        self.gcn1 = GCNConv(in_channels, hidden_channels[0])
        self.gcn2 = GCNConv(hidden_channels[0], hidden_channels[1])
        self.gcn3 = GCNConv(hidden_channels[1], out_channels)
        
        # Dropout layers
        self.dropout = Dropout(dropout_rate)
        
        # Link Prediction MLP Head
        # Concatenated embeddings (2 * out_channels) -> 64 -> 1
        self.mlp_link = torch.nn.Sequential(
            Linear(2 * out_channels, 64),
            torch.nn.ReLU(),
            Dropout(dropout_rate),
            Linear(64, 1)
        )
    
    def encode(self, x, edge_index):
        # GCN Layer 1: 1,433 -> 512
        z = self.gcn1(x, edge_index)
        z = F.relu(z)
        z = self.dropout(z)
        
        # GCN Layer 2: 512 -> 256
        z = self.gcn2(z, edge_index)
        z = F.relu(z)
        z = self.dropout(z)
        
        # GCN Layer 3: 256 -> 128
        z = self.gcn3(z, edge_index)
        
        return z
    
    def decode(self, z, edge_index):
        # Get node pairs for edges
        src, dst = edge_index
        node_pairs = torch.cat([z[src], z[dst]], dim=1)  # [num_edges, 2*128]
        
        # Pass through MLP: 256 -> 64 -> 1
        logits = self.mlp_link(node_pairs).squeeze(1)
        
        return logits
    
    def forward(self, x, edge_index, edge_label_index):
        z = self.encode(x, edge_index)
        logits = self.decode(z, edge_label_index)
        return logits


def train_link_predictor(model, data, optimizer, epoch):
    model.train()
    
    # Get positive edges from training set
    pos_edge_index = data.train_pos_edge_index
    
    # Sample negative edges (same number as positive edges)
    neg_edge_index = negative_sampling(
        edge_index=data.edge_index,
        num_nodes=data.num_nodes,
        num_neg_samples=pos_edge_index.size(1)
    )
    
    # Combine positive and negative edges for training
    edge_label_index = torch.cat([pos_edge_index, neg_edge_index], dim=1)
    edge_label = torch.cat([
        torch.ones(pos_edge_index.size(1)),
        torch.zeros(neg_edge_index.size(1))
    ]).to(data.x.device)
    
    # Forward pass
    logits = model(data.x, data.edge_index, edge_label_index)
    
    # Compute loss
    loss = F.binary_cross_entropy_with_logits(logits, edge_label)
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.item()


@torch.no_grad()
def evaluate_link_predictor(model, data, edge_type='test'):
    model.eval()
    
    # Get positive edges
    if edge_type == 'test':
        pos_edge_index = data.test_pos_edge_index
    else:
        pos_edge_index = data.val_pos_edge_index
    
    # Sample negative edges
    neg_edge_index = negative_sampling(
        edge_index=data.edge_index,
        num_nodes=data.num_nodes,
        num_neg_samples=pos_edge_index.size(1)
    )
    
    # Combine edges
    edge_label_index = torch.cat([pos_edge_index, neg_edge_index], dim=1)
    edge_label = torch.cat([
        torch.ones(pos_edge_index.size(1)),
        torch.zeros(neg_edge_index.size(1))
    ]).to(data.x.device)
    
    # Get predictions
    logits = model(data.x, data.edge_index, edge_label_index)
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).long()
    
    # Move to CPU for sklearn metrics
    preds_np = preds.cpu().numpy()
    labels_np = edge_label.cpu().numpy()
    probs_np = probs.cpu().detach().numpy()
    
    # Compute metrics
    accuracy = accuracy_score(labels_np, preds_np)
    precision = precision_score(labels_np, preds_np, zero_division=0)
    recall = recall_score(labels_np, preds_np, zero_division=0)
    f1 = f1_score(labels_np, preds_np, zero_division=0)
    auc = roc_auc_score(labels_np, probs_np)
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc
    }
    
    return metrics


def main():
    """Main training loop for link prediction"""
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # ========== DATASET LOADING ==========
    print("=" * 70)
    print("LOADING CORA DATASET")
    print("=" * 70)
    
    dataset = Planetoid(root='data/Planetoid', name='Cora', transform=NormalizeFeatures())
    data = dataset[0]
    data = data.to(device)
    
    print(f"\nDataset Statistics:")
    print(f"  - Number of nodes: {data.num_nodes}")
    print(f"  - Number of edges: {data.num_edges}")
    print(f"  - Number of node features: {data.num_node_features}")
    print(f"  - Number of classes: {dataset.num_classes}")
    print(f"  - Node feature shape: {data.x.shape}")
    print(f"  - Edge index shape: {data.edge_index.shape}")
    
    # ========== EDGE TRAIN/TEST SPLIT ==========
    print(f"\n" + "=" * 70)
    print("EDGE TRAIN/TEST SPLIT")
    print("=" * 70)
    
    # Store original edges before split
    original_edge_index = data.edge_index.clone()
    original_num_edges = data.edge_index.size(1)
    
    # Split edges into train/val/test (70%/15%/15%)
    data = train_test_split_edges(data, val_ratio=0.15, test_ratio=0.15)
    
    # Restore full edge_index for GCN encoding
    data.edge_index = original_edge_index
    
    print(f"\n  Training edges (positive): {data.train_pos_edge_index.size(1)}")
    print(f"  Validation edges (positive): {data.val_pos_edge_index.size(1)}")
    print(f"  Test edges (positive): {data.test_pos_edge_index.size(1)}")
    print(f"  Total positive edges: {data.train_pos_edge_index.size(1) + data.val_pos_edge_index.size(1) + data.test_pos_edge_index.size(1)}")
    print(f"  Original edge count: {original_num_edges}")
    
    # ========== MODEL ARCHITECTURE ==========
    print(f"\n" + "=" * 70)
    print("MODEL ARCHITECTURE")
    print("=" * 70)
    
    in_channels = data.num_node_features  # 1,433
    hidden_channels = [512, 256]           # Hidden layer dimensions
    out_channels = 128                      # Embedding dimension
    dropout_rate = 0.5
    
    model = GCNLinkPredictor(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        out_channels=out_channels,
        dropout_rate=dropout_rate
    ).to(device)
    
    print(f"\nGCN Link Predictor Architecture:")
    print(f"  - Input features: {in_channels}")
    print(f"  - GCN Layer 1: {in_channels} -> {hidden_channels[0]}")
    print(f"  - GCN Layer 2: {hidden_channels[0]} -> {hidden_channels[1]}")
    print(f"  - GCN Layer 3: {hidden_channels[1]} -> {out_channels}")
    print(f"  - Link Prediction MLP: {2*out_channels} -> 64 -> 1")
    print(f"  - Dropout rate: {dropout_rate}")
    print(f"  - Activation: ReLU")
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  - Total parameters: {total_params}")
    
    # ========== TRAINING SETUP ==========
    print(f"\n" + "=" * 70)
    print("TRAINING SETUP")
    print("=" * 70)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = F.binary_cross_entropy_with_logits
    num_epochs = 1000
    
    print(f"\n  - Optimizer: Adam")
    print(f"  - Learning rate: 0.01")
    print(f"  - Loss function: Binary Cross Entropy with Logits")
    print(f"  - Epochs: {num_epochs}")
    print(f"  - Batch type: Full batch (all edges)")
    
    # ========== TRAINING LOOP ==========
    print(f"\n" + "=" * 70)
    print("TRAINING PROGRESS")
    print("=" * 70 + "\n")
    
    best_val_auc = 0
    best_epoch = 0
    train_losses = []
    val_accuracies = []
    val_epochs = []
    
    for epoch in range(num_epochs):
        # Train
        loss = train_link_predictor(model, data, optimizer, epoch)
        train_losses.append(loss)
        
        # Validate every 10 epochs
        if (epoch + 1) % 10 == 0:
            val_metrics = evaluate_link_predictor(model, data, edge_type='val')
            val_accuracies.append(val_metrics['accuracy'])
            val_epochs.append(epoch + 1)
            
            print(f"Epoch {epoch+1:3d}/{num_epochs} | Loss: {loss:.4f} | "
                  f"Val Acc: {val_metrics['accuracy']:.4f} | "
                  f"Val AUC: {val_metrics['auc']:.4f}")
            
            # Track best model
            if val_metrics['auc'] > best_val_auc:
                best_val_auc = val_metrics['auc']
                best_epoch = epoch + 1
    
    # ========== FINAL EVALUATION ==========
    print(f"\n" + "=" * 70)
    print("FINAL EVALUATION ON TEST SET")
    print("=" * 70)
    
    test_metrics = evaluate_link_predictor(model, data, edge_type='test')
    
    print(f"\nTest Set Link Prediction Performance:")
    print(f"  - Accuracy:  {test_metrics['accuracy']:.4f}")
    print(f"  - Precision: {test_metrics['precision']:.4f}")
    print(f"  - Recall:    {test_metrics['recall']:.4f}")
    print(f"  - F1-Score:  {test_metrics['f1']:.4f}")
    print(f"  - AUC-ROC:   {test_metrics['auc']:.4f}")
    
    print(f"\nBest validation performance: {best_val_auc:.4f} AUC at epoch {best_epoch}")
    
    # ========== PLOT RESULTS ==========
    print(f"\n" + "=" * 70)
    print("GENERATING PLOTS")
    print("=" * 70)
    
    plt.figure(figsize=(10, 6))
    plt.plot(val_epochs, val_accuracies, marker='o', linewidth=2, markersize=6, color='blue')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Validation Accuracy', fontsize=12)
    plt.title('Link Prediction Validation Accuracy vs Epoch', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('p1q5.png', dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: p1q5.png")
    plt.close()
    
    # ========== SUMMARY ==========
    print(f"\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    print(f"""
RESULTS:
--------
Test Accuracy: {test_metrics['accuracy']:.4f}
Test AUC-ROC:  {test_metrics['auc']:.4f}
Test F1-Score: {test_metrics['f1']:.4f}
""")


if __name__ == "__main__":
    main()
