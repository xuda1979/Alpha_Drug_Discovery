# ai_molecular_dynamics.py

import torch
import torch.nn as nn
import torch.optim as optim

class MolecularDynamicsModel(nn.Module):
    def __init__(self, input_dim):
        super(MolecularDynamicsModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, input_dim)  # Predicts next state
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def predict_molecular_dynamics(current_state, model):
    """
    Predict the next state of the molecular system using the AI model.
    
    Parameters:
    current_state (torch.Tensor): Current state of the molecular system.
    model (MolecularDynamicsModel): Pre-trained molecular dynamics model.
    
    Returns:
    torch.Tensor: Predicted next state.
    """
    return model(current_state)

def train_molecular_dynamics_model(X, y, epochs=100, batch_size=32, lr=0.001):
    model = MolecularDynamicsModel(X.shape[1])
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    dataset = torch.utils.data.TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        if epoch % 10 == 0:
            print(f'Epoch [{epoch}/{epochs}] - Loss: {epoch_loss/len(dataloader)}')

    return model
