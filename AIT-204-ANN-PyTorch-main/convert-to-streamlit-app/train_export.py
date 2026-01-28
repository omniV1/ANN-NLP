# train_export.py
import json, joblib, numpy as np, torch, torch.nn as nn
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

SEED = 42
rng = np.random.default_rng(SEED)

# 1) Data
data = fetch_20newsgroups(subset='all', remove=('headers','footers','quotes'))
X_raw, y = data.data, data.target
num_classes = len(data.target_names)

# 2) Vectorizer (fit once)
vectorizer = TfidfVectorizer(
    max_features=5000, lowercase=True, stop_words='english',
    strip_accents='unicode', token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b"
)
X = vectorizer.fit_transform(X_raw).toarray()

# 3) Split
from sklearn.model_selection import train_test_split
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)

# 4) Model
class NewsMLP(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    def forward(self, x): return self.net(x)

import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = NewsMLP(Xtr.shape[1], num_classes).to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
crit = nn.CrossEntropyLoss()

# 5) Torch training with mini-batches
from torch.utils.data import TensorDataset, DataLoader

Xtr_t = torch.tensor(Xtr, dtype=torch.float32)
ytr_t = torch.tensor(ytr, dtype=torch.long)
Xte_t = torch.tensor(Xte, dtype=torch.float32)
yte_t = torch.tensor(yte, dtype=torch.long)

train_loader = DataLoader(TensorDataset(Xtr_t, ytr_t), batch_size=64, shuffle=True)

for epoch in range(30):
    model.train()
    total_loss = 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss = crit(logits, yb)
        opt.zero_grad(); loss.backward(); opt.step()
        total_loss += loss.item()
    
    # Evaluate every 5 epochs
    if (epoch + 1) % 5 == 0:
        model.eval()
        with torch.no_grad():
            preds = model(Xte_t.to(device)).argmax(dim=1).cpu()
            acc = (preds == yte_t).float().mean().item()
        print(f"Epoch {epoch+1:2d} | Loss: {total_loss/len(train_loader):.4f} | Test Acc: {acc:.2%}")

# 6) Export artifacts
torch.save(model.state_dict(), "model_state_dict.pt")
joblib.dump(vectorizer, "vectorizer.pkl")
with open("label_names.json","w") as f: json.dump(data.target_names, f)
print("Exported: model_state_dict.pt, vectorizer.pkl, label_names.json")
