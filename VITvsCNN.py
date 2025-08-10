import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
import timm
from tqdm import tqdm


transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])
train_data = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_data = torchvision.datasets.CIFAR10(root='./data', train=False, transform=transform)
train_loader = DataLoader(train_data, batch_size=128, shuffle=True)
test_loader = DataLoader(test_data, batch_size=128)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


cnn = torchvision.models.resnet18(pretrained=True)
cnn.fc = torch.nn.Linear(512, 10)
cnn = cnn.to(device)


vit = timm.create_model('vit_tiny_patch16_224', pretrained=False, num_classes=10, img_size=32)
vit = vit.to(device)


def train_model(model, name):
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1)
    criterion = torch.nn.CrossEntropyLoss()
    
    for epoch in range(1):
        model.train()
        for x, y in tqdm(train_loader, desc=f'{name} Epoch {epoch+1}'):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
        scheduler.step()


def evaluate(model):
    model.eval()
    correct = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            correct += (outputs.argmax(1) == y).sum().item()
    return correct / len(test_data)


train_model(cnn, 'CNN')
train_model(vit, 'ViT')
print(f'CNN Accuracy: {evaluate(cnn):.4f}')
print(f'ViT Accuracy: {evaluate(vit):.4f}')