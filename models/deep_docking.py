import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class DockingCNN(nn.Module):
    def __init__(self, input_channels=1):
        super().__init__()
        self.conv1 = nn.Conv3d(input_channels, 8, kernel_size=3, padding=1)
        self.pool = nn.MaxPool3d(2)
        self.fc = nn.Linear(8 * 4 * 4 * 4, 1)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = torch.sigmoid(self.fc(x))
        return x

def train_docking_model(X, y, epochs=5, lr=0.001, batch_size=8):
    """Train a simple 3D CNN for docking score prediction."""
    model = DockingCNN(X.shape[1])
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    dataset = TensorDataset(torch.tensor(X, dtype=torch.float32),
                            torch.tensor(y, dtype=torch.float32))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_x, batch_y in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs.squeeze(), batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        if epoch % 1 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(dataloader)}")
    return model
