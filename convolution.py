import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.datasets import load_digits
from typing import Literal
import matplotlib.pyplot as plt

digits = load_digits(return_X_y=True)[0]
abs_norm = digits.max()
digits = (digits / abs_norm).reshape(digits.shape[0], 1, 8, 8)
dataset = TensorDataset(torch.as_tensor(digits, dtype=torch.float32))
data = DataLoader(dataset, batch_size=100, pin_memory=True, num_workers=0)

class ELBOLoss(nn.Module):
    def __init__(self, reduction: Literal['mean', 'sum'] = 'mean'):
        super(ELBOLoss, self).__init__()
        self.reduction = reduction
        self.reconstruction_loss = nn.BCELoss(reduction=self.reduction)
        self.kl_loss = KullbackLeiblerLoss(reduction=self.reduction)

    def forward(self, X_reconstructed: torch.Tensor, X_origin: torch.Tensor, mu: torch.Tensor, log_var: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        recon_loss = self.reconstruction_loss(X_reconstructed, X_origin)
        kl_loss = self.kl_loss(log_var, mu)
        loss = recon_loss + kl_loss
        return (loss, recon_loss, kl_loss)

class KullbackLeiblerLoss(nn.Module):
    def __init__(self, reduction: Literal['mean', 'sum'] = 'mean'):
        super(KullbackLeiblerLoss, self).__init__()
        self.reduction = reduction

    def forward(self, log_var: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
        losses = -0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1)
        return (torch.mean(losses, dim=0) if self.reduction == 'mean' else torch.sum(losses, dim=0))

class Conv(nn.Module):
    def __init__(self, latent_dim: int, input_shape: int, device: str = 'auto'):
        super(Conv, self).__init__()

        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        self._conv_size = 2

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=2, kernel_size=1),
            nn.ELU(),
            nn.BatchNorm2d(num_features=2),
            nn.Conv2d(in_channels=2, out_channels=4, kernel_size=3),
            nn.ELU(),
            nn.BatchNorm2d(num_features=4),
            nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, stride=2),
            nn.ELU(),
            nn.BatchNorm2d(num_features=8),
            nn.Flatten()
        ).to(self.device)

        self.latent_space_mu = nn.Linear(self._conv_size ** 5, latent_dim).to(self.device)
        self.latent_space_var = nn.Linear(self._conv_size ** 5, latent_dim).to(self.device)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=8, out_channels=4, kernel_size=3),
            nn.ELU(),
            nn.BatchNorm2d(num_features=4),
            nn.ConvTranspose2d(in_channels=4, out_channels=2, kernel_size=3),
            nn.ELU(),
            nn.BatchNorm2d(num_features=2),
            nn.ConvTranspose2d(in_channels=2, out_channels=1, kernel_size=3),
            nn.BatchNorm2d(num_features=1),
            nn.Sigmoid()
        ).to(self.device)

    def encode(self, X: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.encoder(X)
        mu = self.latent_space_mu(encoded)
        log_var = self.latent_space_var(encoded)
        return (mu, log_var)

    def reparametize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        noise = torch.randn_like(std).to(self.device)
        z = mu + noise * std
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        theta = nn.functional.softmax(z, dim=1)
        theta = theta.reshape(shape=(theta.size(0), 8, self._conv_size, self._conv_size))
        return self.decoder(theta)

    def forward(self, X: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, log_var = self.encode(X)
        z = self.reparametize(mu, log_var)
        X_reconst = self.decode(z)
        return (X_reconst, mu, log_var)

    def generate(self, z: torch.Tensor) -> torch.Tensor:
        return self.decode(z).detach()

generator = Conv(input_shape=8, latent_dim=32)
loss_fn = ELBOLoss()
optimizer = torch.optim.Adam(generator.parameters(), lr=2e-3)
epoch_losses = []
for epoch in range(1, 101):
    generator.train()
    running_loss = 0.0
    for X, in data:
        X = X.to(device='cpu')
        X_reconst, mu, log_var = generator(X)
        loss, recon_loss, kl_loss = loss_fn(X_reconst, X, mu, log_var)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        epoch_loss = running_loss / len(data)
        epoch_losses.append(epoch_loss) 
    print(f'epochs {epoch} train loss: {running_loss / len(data)}')

z = torch.normal(0, 1, size=(1, 32), device=generator.device)
generated = generator.generate(z).cpu()
plt.imshow(generated[0, 0] * abs_norm, cmap='gray')
plt.show()

