# protein_structure_prediction.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class SimpleProteinModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SimpleProteinModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, output_dim)  # Predicts structure classes
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return self.fc4(x)

def train_protein_model(X, y, epochs=100, batch_size=32, lr=0.001):
    y_tensor = torch.as_tensor(y)
    if y_tensor.ndim > 1 and y_tensor.size(-1) > 1:
        # Convert one-hot / probability vectors to class indices expected by CrossEntropyLoss
        targets = torch.argmax(y_tensor, dim=1).long()
        output_dim = y_tensor.size(-1)
    else:
        targets = y_tensor.long().view(-1)
        output_dim = int(targets.max().item()) + 1 if targets.numel() > 0 else 1

    model = SimpleProteinModel(X.shape[1], output_dim)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    dataset = torch.utils.data.TensorDataset(torch.tensor(X, dtype=torch.float32), targets)
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

def predict_structure(sequence, model):
    """
    Predict the structure of a protein sequence.

    Parameters:
    sequence (str): The amino acid sequence of the protein.
    model (SimpleProteinModel): The trained model for structure prediction.

    Returns:
    torch.Tensor: Predicted structure classes.
    """
    # Convert sequence to one-hot encoding or other suitable format
    model.eval()
    sequence_encoded = encode_sequence(sequence, model.fc1.in_features)
    with torch.no_grad():
        logits = model(sequence_encoded)
    return torch.softmax(logits, dim=-1)

def encode_sequence(sequence, expected_dim):
    amino_acids = "ACDEFGHIKLMNPQRSTVWY"
    encoding = torch.zeros(len(sequence), len(amino_acids), dtype=torch.float32)
    aa_to_idx = {aa: i for i, aa in enumerate(amino_acids)}
    for i, aa in enumerate(sequence):
        idx = aa_to_idx.get(aa, None)
        if idx is not None:
            encoding[i, idx] = 1.0
    flat = encoding.reshape(-1)
    if flat.numel() < expected_dim:
        flat = F.pad(flat, (0, expected_dim - flat.numel()))
    elif flat.numel() > expected_dim:
        flat = flat[:expected_dim]
    return flat.unsqueeze(0)
