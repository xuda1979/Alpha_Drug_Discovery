# adversarial_toxicity.py

import torch
import torch.nn as nn
import torch.optim as optim

class ToxicityDiscriminator(nn.Module):
    def __init__(self, input_dim):
        super(ToxicityDiscriminator, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x

class ToxicityGenerator(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super(ToxicityGenerator, self).__init__()
        self.fc1 = nn.Linear(latent_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_dim)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        return x

def train_toxicity_gan(X, latent_dim=100, epochs=1000, batch_size=64, lr=0.0002):
    generator = ToxicityGenerator(latent_dim, X.shape[1])
    discriminator = ToxicityDiscriminator(X.shape[1])

    criterion = nn.BCELoss()
    optimizer_g = optim.Adam(generator.parameters(), lr=lr)
    optimizer_d = optim.Adam(discriminator.parameters(), lr=lr)

    real_labels = torch.ones(batch_size, 1)
    fake_labels = torch.zeros(batch_size, 1)

    for epoch in range(epochs):
        # Train Discriminator
        discriminator.zero_grad()
        real_data = torch.tensor(X, dtype=torch.float32)[:batch_size]
        output_real = discriminator(real_data)
        loss_real = criterion(output_real, real_labels)
        loss_real.backward()

        noise = torch.randn(batch_size, latent_dim)
        fake_data = generator(noise)
        output_fake = discriminator(fake_data.detach())
        loss_fake = criterion(output_fake, fake_labels)
        loss_fake.backward()
        optimizer_d.step()

        # Train Generator
        generator.zero_grad()
        output_fake = discriminator(fake_data)
        loss_g = criterion(output_fake, real_labels)
        loss_g.backward()
        optimizer_g.step()

        if epoch % 100 == 0:
            print(f'Epoch [{epoch}/{epochs}] - Loss D: {loss_real + loss_fake}, Loss G: {loss_g}')

    return generator, discriminator
