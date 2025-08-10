import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import re
import math
from datasets import load_dataset

dataset = load_dataset("imdb")
train_data = dataset["train"]
test_data = dataset["test"]
#print(train_data[0])

def tokenize(text):
    return re.findall(r'\w+|[^\w\s]', text.lower())
word_counts = Counter()
for example in train_data:
    word_counts.update(tokenize(example["text"]))

vocab = {'<PAD>': 0, '<UNK>': 1}
for word, _ in word_counts.most_common(10000):
    vocab[word] = len(vocab)


class IMDBDataset(Dataset):
    def __init__(self, hf_dataset, vocab, max_len=256):
        self.texts = []
        self.labels = []
        for example in hf_dataset:
            tokens = tokenize(example["text"])
            self.texts.append(torch.tensor([vocab.get(t, 1) for t in tokens]))
            self.labels.append(example["label"])
        self.labels = torch.tensor(self.labels)
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx][:self.max_len]
        pad_len = self.max_len - len(text)
        padded = torch.cat([text, torch.zeros(pad_len, dtype=torch.long)])
        return padded, self.labels[idx]

train_dataset = IMDBDataset(train_data, vocab)
test_dataset = IMDBDataset(test_data, vocab)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16)



dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=512, n_heads=8):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
    def forward(self, x):
        batch_size = x.size(0)

        Q = self.W_q(x).view(batch_size, -1, self.n_heads, self.d_head).transpose(1, 2) 
        K = self.W_k(x).view(batch_size, -1, self.n_heads, self.d_head).transpose(1, 2)
        V = self.W_v(x).view(batch_size, -1, self.n_heads, self.d_head).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_head ** 0.5) 
        attn = torch.softmax(scores, dim=-1)
        output = torch.matmul(attn, V) 

        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.W_o(output)

class TransformerBlock(nn.Module):
    def __init__(self, d_model=512, n_heads=8, ff_dim=2048, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        attn_out = self.attention(x)
        x = self.norm1(x + self.dropout(attn_out))
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout(ff_out))
        return x

class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size=10000, d_model=512, n_heads=8, num_layers=6, num_classes=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model) 
        self.layers = nn.ModuleList([TransformerBlock(d_model, n_heads) for _ in range(num_layers)])
        self.fc = nn.Linear(d_model, num_classes)
        
    def forward(self, x):
        x = self.embedding(x) 
        x = self.pos_encoding(x)
        for layer in self.layers:
            x = layer(x)
        x = x.mean(dim=1) 
        return self.fc(x)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TransformerClassifier(len(vocab)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

for epoch in range(5): 
    model.train()
    total_loss = 0
    for texts, labels in train_loader:
        texts, labels = texts.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(texts)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch: {epoch}, Loss: {total_loss / len(train_loader)}")

def predict(text):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for texts, labels in test_loader:
            texts, labels = texts.to(device), labels.to(device)
            outputs = model(texts)
            predicted = outputs.argmax(dim=1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    print(f"Accuracy: {correct / total * 100:.2f}%")

def predict_sentiment(text, model, vocab):
    model.eval()
    tokens = tokenize(text)
    input_ids = torch.tensor([vocab.get(t, 1) for t in tokens]).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(input_ids)
    prob = torch.softmax(output, dim=1)
    return "Positive" if prob.argmax().item() == 1 else "Negative"


print(predict_sentiment("This movie is a masterpiece!", model, vocab)) 
print(predict_sentiment("Waste of time.", model, vocab)) 

