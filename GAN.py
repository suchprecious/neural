import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
import numpy as np
import matplotlib.pyplot as plt
from torchmetrics.image.fid import FrechetInceptionDistance

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 128
latent_dim = 100
lr = 0.001
beta1 = 0.5
beta2 = 0.999
epochs = 10

transform = transforms.Compose([
    transforms.Resize(64),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

n = 1000
indices = torch.arange(n)
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
train_subset = torch.utils.data.Subset(train_dataset, indices)
train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)

class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.query = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.key = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.value = nn.Conv2d(in_channels, in_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        B, C, H, W = x.shape
        q = self.query(x).view(B, -1, H * W).permute(0, 2, 1)
        k = self.key(x).view(B, -1, H * W)
        v = self.value(x).view(B, -1, H * W)
        
        attention = torch.softmax(torch.bmm(q, k), dim=-1)
        out = torch.bmm(v, attention.permute(0, 2, 1))
        out = out.view(B, C, H, W)
        return self.gamma * out + x

class Generator(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.main = nn.Sequential( 
            nn.ConvTranspose2d(latent_dim, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            
            
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
                        
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
                        
            SelfAttention(128),            
            
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
                        
            nn.ConvTranspose2d(64, 1, 4, 2, 1, bias=False),
            nn.Tanh()
        )
        
    def forward(self, x):
        x = x.view(-1, latent_dim, 1, 1)
        return self.main(x)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(1, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            
            SelfAttention(128),
            
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            
            
            nn.Conv2d(512, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        x = self.main(x)
        return x.view(-1, 1)

generator = Generator(latent_dim).to(device)
discriminator = Discriminator().to(device)

criterion = nn.BCELoss()
optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, beta2))
optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, beta2))

fid = FrechetInceptionDistance(feature=64, normalize=True).to(device)

def prepare_images_for_fid(images):
   
    images = (images + 1) / 2
    
    images = (images * 255).type(torch.uint8)
    
    if images.shape[1] == 1:
        images = images.repeat(1, 3, 1, 1)
    return images

for epoch in range(epochs):
    for i, (real_images, _) in enumerate(train_loader):
        real_images = real_images.to(device)
        batch_size = real_images.size(0)
        
        real_labels = torch.ones(batch_size, 1).to(device)
        fake_labels = torch.zeros(batch_size, 1).to(device)
        
        optimizer_D.zero_grad()
        
        real_outputs = discriminator(real_images)
        d_loss_real = criterion(real_outputs, real_labels)
        
        z = torch.randn(batch_size, latent_dim).to(device)
        fake_images = generator(z)
        fake_outputs = discriminator(fake_images.detach())
        d_loss_fake = criterion(fake_outputs, fake_labels)
        
        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        optimizer_D.step()
        
        optimizer_G.zero_grad()
        
        fake_outputs = discriminator(fake_images)
        g_loss = criterion(fake_outputs, real_labels)
        g_loss.backward()
        optimizer_G.step()
        
        if i % 100 == 0:
            print(f"Epoch [{epoch + 1}/{epochs}]"
                  f"D_loss: {d_loss.item():.4f}, G_loss: {g_loss.item():.4f}")
                
            with torch.no_grad():
                test_z = torch.randn(16, latent_dim).to(device)
                generated = generator(test_z)
                save_image(generated, f"C:/Users/Павел/Desktop/folders/neural/images/generated_epoch_{epoch}_batch_{i}.png", nrow=4, normalize=True)
                  
                real_for_fid = prepare_images_for_fid(real_images[:16])
                fake_for_fid = prepare_images_for_fid(generated)
               
                fid.update(real_for_fid, real=True)
                fid.update(fake_for_fid, real=False)
                print(f"FID: {fid.compute().item():.4f}")
                fid.reset()


torch.save(generator.state_dict(), 'generator.pth')
torch.save(discriminator.state_dict(), 'discriminator.pth')