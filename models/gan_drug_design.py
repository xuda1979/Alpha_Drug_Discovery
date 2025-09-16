# gan_drug_design.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class Generator(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(latent_dim, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 512)
        self.fc4 = nn.Linear(512, output_dim)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.tanh(self.fc4(x))
        return x

class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 1)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.sigmoid(self.fc4(x))
        return x

def train_gan(X, latent_dim=100, epochs=1000, batch_size=64, lr=0.0002):
    if len(X) == 0:
        raise ValueError("Input dataset X must contain at least one sample")

    generator = Generator(latent_dim, X.shape[1])
    discriminator = Discriminator(X.shape[1])

    criterion = nn.BCELoss()
    optimizer_g = optim.Adam(generator.parameters(), lr=lr)
    optimizer_d = optim.Adam(discriminator.parameters(), lr=lr)

    dataset = TensorDataset(torch.tensor(X, dtype=torch.float32))
    dataloader = DataLoader(dataset, batch_size=min(batch_size, len(dataset)), shuffle=True)

    for epoch in range(epochs):
        epoch_loss_d = 0.0
        epoch_loss_g = 0.0

        for (real_data,) in dataloader:
            current_batch_size = real_data.size(0)

            # Train Discriminator
            optimizer_d.zero_grad()
            real_labels = torch.ones(current_batch_size, 1, dtype=real_data.dtype)
            fake_labels = torch.zeros(current_batch_size, 1, dtype=real_data.dtype)

            output_real = discriminator(real_data)
            loss_real = criterion(output_real, real_labels)

            noise = torch.randn(current_batch_size, latent_dim)
            fake_data = generator(noise)
            output_fake = discriminator(fake_data.detach())
            loss_fake = criterion(output_fake, fake_labels)

            loss_d = loss_real + loss_fake
            loss_d.backward()
            optimizer_d.step()

            # Train Generator
            optimizer_g.zero_grad()
            output_fake = discriminator(fake_data)
            loss_g = criterion(output_fake, real_labels)
            loss_g.backward()
            optimizer_g.step()

            epoch_loss_d += loss_d.item()
            epoch_loss_g += loss_g.item()

        if epoch % 100 == 0:
            mean_loss_d = epoch_loss_d / len(dataloader)
            mean_loss_g = epoch_loss_g / len(dataloader)
            print(f'Epoch [{epoch}/{epochs}] - Loss D: {mean_loss_d:.4f}, Loss G: {mean_loss_g:.4f}')

    return generator
