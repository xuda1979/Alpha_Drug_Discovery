# generative_model.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc21 = nn.Linear(128, latent_dim)  # mean
        self.fc22 = nn.Linear(128, latent_dim)  # log variance
        self.fc3 = nn.Linear(latent_dim, 128)
        self.fc4 = nn.Linear(128, input_dim)

    def encode(self, x):
        h1 = torch.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = torch.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

def vae_loss(recon_x, x, mu, logvar):
    BCE = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

def train_vae(X, latent_dim=20, epochs=10, learning_rate=0.001, batch_size=32):
    model = VAE(X.shape[1], latent_dim)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    dataset = TensorDataset(torch.tensor(X, dtype=torch.float32))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for batch_X, in dataloader:
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(batch_X)
            loss = vae_loss(recon_batch, batch_X, mu, logvar)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(dataloader)}")

    return model

def generate_new_molecules(model, num_samples=10):
    model.eval()
    z = torch.randn(num_samples, model.fc21.out_features)
    generated = model.decode(z)
    return generated.detach().numpy()
