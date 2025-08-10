import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

dataset = load_dataset("imdb")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512, return_tensors="pt")

tokenized_datasets = dataset.map(tokenize_function, batched=True)
tokenized_datasets = tokenized_datasets.remove_columns(["text"])
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")

train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(100))
eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(10))

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert self.head_dim * num_heads == embed_dim, "Embed dim must be divisible by num_heads"
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
    
    def forward(self, x):
        batch_size, seq_len, _ = x.size()

        Q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, V)

        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.out_proj(attn_output)

class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=512):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim))
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(1)]

class TransformerEncoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, hidden_dim, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(embed_dim, num_heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim)
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        attn_output = self.attention(x)
        x = self.norm1(x + self.dropout(attn_output))
        
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_output))
        return x


class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, hidden_dim, num_layers, num_classes, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoding = PositionalEncoding(embed_dim)
        self.layers = nn.ModuleList([
            TransformerEncoderBlock(embed_dim, num_heads, hidden_dim, dropout)
            for _ in range(num_layers)
        ])
        self.classifier = nn.Linear(embed_dim, num_classes)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, input_ids):
        x = self.embedding(input_ids)
        x = self.pos_encoding(x)
        x = self.dropout(x)

        for layer in self.layers:
            x = layer(x)

        pooled = x.mean(dim=1)
        return self.classifier(pooled)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TransformerClassifier(
    vocab_size=tokenizer.vocab_size,
    embed_dim=256,
    num_heads=8,
    hidden_dim=512,
    num_layers=6,
    num_classes=2
).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

def collate_fn(batch):
    return {
        "input_ids": torch.stack([torch.tensor(item["input_ids"]) for item in batch]),
        "labels": torch.tensor([item["labels"] for item in batch])
    }

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)
eval_loader = DataLoader(eval_dataset, batch_size=16, collate_fn=collate_fn)

# for epoch in range(5):
#     model.train()
#     total_loss = 0
#     for batch in train_loader:
#         inputs = batch["input_ids"].to(device)
#         labels = batch["labels"].to(device)
        
#         optimizer.zero_grad()
#         outputs = model(inputs)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()
        
#         total_loss += loss.item()
    
#     print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_loader)}")

train_losses = []
eval_losses = []
accuracies = []

for epoch in range(5):
    model.train()
    total_loss = 0
    for batch in train_loader:
        inputs = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    avg_train_loss = total_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    model.eval()
    eval_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in eval_loader:
            inputs = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(inputs)
            
            eval_loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_eval_loss = eval_loss / len(eval_loader)
    accuracy = 100 * correct / total
    
    eval_losses.append(avg_eval_loss)
    accuracies.append(accuracy)
    
    print(f"Epoch {epoch+1}/{5}")
    print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_eval_loss:.4f}")
    print(f"Accuracy: {accuracy:.2f}%")
    print("-" * 50)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label="Train Loss", marker="o")
plt.plot(eval_losses, label="Val Loss", marker="o")
plt.title("Training and Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.subplot(1, 2, 2)
plt.plot(accuracies, label="Accuracy", marker="o", color="green")
plt.title("Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.ylim(0, 100)
plt.grid(True)
plt.tight_layout()
plt.show()

cm = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Negative", "Positive"])
disp.plot(cmap="Blues")
plt.title("Confusion Matrix")
plt.show()

def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = model(inputs["input_ids"])
        probs = F.softmax(outputs, dim=1)
        pred = torch.argmax(probs).item()
    
    sentiment = "Positive" if pred == 1 else "Negative"
    confidence = probs[0][pred].item() * 100
    print(f"Text: {text[:100]}...")
    print(f"Predicted: {sentiment} | Confidence: {confidence:.2f}%")
    print("-" * 50)

test_samples = [
    "This movie was bad!",
    "I hated every minute of this film.",
    "It was awful!"
]

for sample in test_samples:
    predict_sentiment(sample)