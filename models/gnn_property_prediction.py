"""Graph neural network for molecular property prediction."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


class GCNLayer(nn.Module):
    """Simple graph convolution layer."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """Apply a single GCN layer with symmetric adjacency normalisation."""

        if adj.dim() != 2:
            raise ValueError("Adjacency matrix must be 2-dimensional")

        # Add self-loops before computing the degree matrix.
        device = x.device
        adj_with_self_loops = adj + torch.eye(adj.size(0), device=device, dtype=adj.dtype)
        degree = adj_with_self_loops.sum(dim=1)
        deg_inv_sqrt = degree.pow(-0.5)
        deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0
        norm_adj = deg_inv_sqrt.unsqueeze(1) * adj_with_self_loops * deg_inv_sqrt.unsqueeze(0)

        support = self.linear(x)
        out = torch.matmul(norm_adj, support)
        return F.relu(out)


class GCN(nn.Module):
    """Two-layer GCN for graph-level binary classification."""

    def __init__(self, num_features: int, hidden_dim: int = 16):
        super().__init__()
        self.conv1 = GCNLayer(num_features, hidden_dim)
        self.conv2 = GCNLayer(hidden_dim, 1)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x, adj)
        h = self.conv2(h, adj)
        # Global mean pooling retaining batch dimension
        return torch.sigmoid(h.mean(dim=0))


class GraphDataset(Dataset):
    """Dataset of graphs represented by feature and adjacency matrices."""

    def __init__(self, features, adjs, labels):
        self.features = [torch.tensor(f, dtype=torch.float32) for f in features]
        self.adjs = [torch.tensor(a, dtype=torch.float32) for a in adjs]
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        return self.features[idx], self.adjs[idx], self.labels[idx]


def train_gcn(features, adjs, labels, epochs: int = 5, lr: float = 0.01) -> GCN:
    """Train a small graph convolutional network."""
    dataset = GraphDataset(features, adjs, labels)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    model = GCN(features[0].shape[1])
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    for epoch in range(epochs):
        epoch_loss = 0.0
        for x, adj, label in dataloader:
            optimizer.zero_grad()
            graph_x = x.squeeze(0)
            graph_adj = adj.squeeze(0)
            output = model(graph_x, graph_adj)
            loss = criterion(output.view(1), label.float().view(1))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(dataloader):.4f}")

    return model
