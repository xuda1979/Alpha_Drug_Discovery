import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class DrugTargetModel(nn.Module):
    def __init__(self, input_dim):
        super(DrugTargetModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x

def train_drug_target_model(X, y, epochs=10, learning_rate=0.001, batch_size=32, save_path=None):
    """
    Train a drug-target interaction model.

    Parameters:
    X (np.ndarray): Input features.
    y (np.ndarray): Target values.
    epochs (int): Number of training epochs.
    learning_rate (float): Learning rate for the optimizer.
    batch_size (int): Size of each mini-batch.
    save_path (str): Path to save the trained model.

    Returns:
    DrugTargetModel: Trained model.
    """
    model = DrugTargetModel(X.shape[1])
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    dataset = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs.squeeze(), batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(dataloader)}")

    if save_path:
        torch.save(model.state_dict(), save_path)
        print(f"Model saved to {save_path}")

    return model
