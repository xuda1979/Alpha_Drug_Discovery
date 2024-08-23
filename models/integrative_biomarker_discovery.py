# integrative_biomarker_discovery.py

import torch
import torch.nn as nn
import torch.optim as optim

class IntegrativeBiomarkerModel(nn.Module):
    def __init__(self, genomics_dim, proteomics_dim, metabolomics_dim):
        super(IntegrativeBiomarkerModel, self).__init__()
        self.genomics_fc = nn.Linear(genomics_dim, 128)
        self.proteomics_fc = nn.Linear(proteomics_dim, 128)
        self.metabolomics_fc = nn.Linear(metabolomics_dim, 128)
        self.fc_combined = nn.Linear(128 * 3, 64)
        self.fc_output = nn.Linear(64, 1)
    
    def forward(self, genomics, proteomics, metabolomics):
        genomics_out = torch.relu(self.genomics_fc(genomics))
        proteomics_out = torch.relu(self.proteomics_fc(proteomics))
        metabolomics_out = torch.relu(self.metabolomics_fc(metabolomics))
        combined = torch.cat([genomics_out, proteomics_out, metabolomics_out], dim=1)
        combined_out = torch.relu(self.fc_combined(combined))
        output = torch.sigmoid(self.fc_output(combined_out))
        return output

def train_integrative_biomarker_model(X_genomics, X_proteomics, X_metabolomics, y, epochs=100, batch_size=32, lr=0.001):
    model = IntegrativeBiomarkerModel(X_genomics.shape[1], X_proteomics.shape[1], X_metabolomics.shape[1])
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    dataset = torch.utils.data.TensorDataset(
        torch.tensor(X_genomics, dtype=torch.float32),
        torch.tensor(X_proteomics, dtype=torch.float32),
        torch.tensor(X_metabolomics, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32)
    )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for batch_genomics, batch_proteomics, batch_metabolomics, batch_y in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_genomics, batch_proteomics, batch_metabolomics)
            loss = criterion(outputs.squeeze(), batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        if epoch % 10 == 0:
            print(f'Epoch [{epoch}/{epochs}] - Loss: {epoch_loss/len(dataloader)}')

    return model

