"""Graph neural network for molecular property prediction."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


class GCNLayer(nn.Module):
    """Simple graph convolution layer."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        out = torch.matmul(adj, x)
        out = self.linear(out)
        return torch.relu(out)


class GCN(nn.Module):
    """Two-layer GCN for graph-level binary classification."""

    def __init__(self, num_features: int, hidden_dim: int = 16):
        super().__init__()
        self.conv1 = GCNLayer(num_features, hidden_dim)
        self.conv2 = GCNLayer(hidden_dim, 1)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x, adj)
        h = self.conv2(h, adj)
        # Global mean pooling
        return torch.sigmoid(h.mean())


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
            output = model(x.squeeze(0), adj.squeeze(0))
            loss = criterion(output.squeeze(), label)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(dataloader):.4f}")

    return model
