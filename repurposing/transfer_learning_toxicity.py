# transfer_learning_toxicity.py

import torch
import torch.nn as nn
import torch.optim as optim

class PretrainedToxicityModel(nn.Module):
    def __init__(self, input_dim, pretrained_model_path):
        super(PretrainedToxicityModel, self).__init__()
        self.pretrained_model = torch.load(pretrained_model_path)
        self.fc = nn.Linear(input_dim, 1)
    
    def forward(self, x):
        x = self.pretrained_model(x)
        x = torch.sigmoid(self.fc(x))
        return x

def fine_tune_toxicity_model(X, y, pretrained_model_path, epochs=100, batch_size=32, lr=0.001):
    model = PretrainedToxicityModel(X.shape[1], pretrained_model_path)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    dataset = torch.utils.data.TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

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

        if epoch % 10 == 0:
            print(f'Epoch [{epoch}/{epochs}] - Loss: {epoch_loss/len(dataloader)}')

    return model
