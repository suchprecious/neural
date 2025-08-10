import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

dataset = Planetoid(root='data/Cora', name='Cora', transform=NormalizeFeatures())
data = dataset[0]

train_mask = data.train_mask
test_mask = data.test_mask

class GCN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.conv1 = GCNConv(dataset.num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, dataset.num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x

class GAT(torch.nn.Module):
    def __init__(self, hidden_channels, heads=1):
        super().__init__()
        self.conv1 = GATConv(dataset.num_features, hidden_channels, heads=heads)
        self.conv2 = GATConv(hidden_channels * heads, dataset.num_classes, heads=1)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x

def train_test_model(model, data, epochs=200):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = criterion(out[train_mask], data.y[train_mask])
        loss.backward()
        optimizer.step()
        
    model.eval()
    with torch.no_grad():
        pred = model(data.x, data.edge_index).argmax(dim=1)
        acc = accuracy_score(data.y[test_mask].cpu(), pred[test_mask].cpu())
    return acc, pred

gcn_model = GCN(hidden_channels=16)
gat_model = GAT(hidden_channels=8, heads=8) 

gcn_acc, gcn_pred = train_test_model(gcn_model, data)
gat_acc, gat_pred = train_test_model(gat_model, data)

print(f'GCN Accuracy: {gcn_acc:.4f}')
print(f'GAT Accuracy: {gat_acc:.4f}')

rf = RandomForestClassifier(n_estimators=100)
rf.fit(data.x[train_mask].cpu().numpy(), data.y[train_mask].cpu().numpy())
rf_pred = rf.predict(data.x[test_mask].cpu().numpy())
rf_acc = accuracy_score(data.y[test_mask].cpu(), rf_pred)
print(f'RF Accuracy: {rf_acc:.4f}')

def visualize_embeddings(model, data):
    model.eval()
    with torch.no_grad():
        out = model.conv1(data.x, data.edge_index)
        embeddings = out.cpu().numpy()
    
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)
    
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                          c=data.y.cpu(), cmap='Set1', alpha=0.7)
    plt.legend(*scatter.legend_elements(), title="Classes")
    plt.title(f'{model.__class__.__name__} Embeddings Visualization')
    plt.show()

visualize_embeddings(gcn_model, data)
visualize_embeddings(gat_model, data)