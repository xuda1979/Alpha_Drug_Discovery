# multiomics_integration.py

import torch
import torch.nn as nn
import torch.optim as optim

class MultiModalNet(nn.Module):
    def __init__(self, input_dims, output_dim):
        super(MultiModalNet, self).__init__()
        self.fc_genomics = nn.Linear(input_dims['genomics'], 128)
        self.fc_proteomics = nn.Linear(input_dims['proteomics'], 128)
        self.fc_transcriptomics = nn.Linear(input_dims['transcriptomics'], 128)
        self.fc_combined = nn.Linear(128 * 3, 64)
        self.fc_output = nn.Linear(64, output_dim)

    def forward(self, genomics, proteomics, transcriptomics):
        genomics_out = torch.relu(self.fc_genomics(genomics))
        proteomics_out = torch.relu(self.fc_proteomics(proteomics))
        transcriptomics_out = torch.relu(self.fc_transcriptomics(transcriptomics))
        
        combined = torch.cat((genomics_out, proteomics_out, transcriptomics_out), dim=1)
        combined_out = torch.relu(self.fc_combined(combined))
        output = self.fc_output(combined_out)
        return output

def train_multimodal_model(X_genomics, X_proteomics, X_transcriptomics, y, epochs=10, learning_rate=0.001, batch_size=32):
    input_dims = {
        'genomics': X_genomics.shape[1],
        'proteomics': X_proteomics.shape[1],
        'transcriptomics': X_transcriptomics.shape[1]
    }
    model = MultiModalNet(input_dims, y.shape[1])
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    dataset = TensorDataset(torch.tensor(X_genomics, dtype=torch.float32),
                            torch.tensor(X_proteomics, dtype=torch.float32),
                            torch.tensor(X_transcriptomics, dtype=torch.float32),
                            torch.tensor(y, dtype=torch.float32))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for batch_genomics, batch_proteomics, batch_transcriptomics, batch_y in dataloader:
            optimizer.zero_grad()
            output = model(batch_genomics, batch_proteomics, batch_transcriptomics)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(dataloader)}")

    return model
