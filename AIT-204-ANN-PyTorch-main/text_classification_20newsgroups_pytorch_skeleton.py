# text_classification_20newsgroups_pytorch_skeleton.py
# Purpose: TF-IDF + PyTorch MLP for 20 Newsgroups Classification
# Maximum accuracy version with optimized hyperparameters

import os
import random
import numpy as np

# ---- Reproducibility ----
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)

import torch
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

# =========================
# Hyperparameters (Optimized)
# =========================
MAX_FEATURES = 10000      # More features = more information
HIDDEN_1 = 512            # Larger first layer
HIDDEN_2 = 256            # Second layer
HIDDEN_3 = 128            # Third layer for depth
DROPOUT = 0.4             # Balanced dropout
WEIGHT_DECAY = 5e-4       # L2 regularization
LEARNING_RATE = 2e-3      # Slightly higher LR
BATCH_SIZE = 32           # Smaller batches = more updates, better generalization
MAX_EPOCHS = 50           # More room to train
PATIENCE = 7              # More patience for convergence
LABEL_SMOOTHING = 0.1     # Prevents overconfident predictions

# =========================
# Load Dataset
# =========================
print("Loading 20 Newsgroups dataset...")
data = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
X_raw, y = data.data, data.target
num_classes = len(data.target_names)
print(f"Loaded {len(X_raw)} documents across {num_classes} categories")

# =========================
# Convert Text Data to Numerical Format (TF-IDF)
# =========================
print(f"\nVectorizing text with TF-IDF (max_features={MAX_FEATURES})...")
vectorizer = TfidfVectorizer(
    max_features=MAX_FEATURES,
    lowercase=True,
    stop_words='english',
    strip_accents='unicode',
    token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b",
    ngram_range=(1, 2),    # Include bigrams for better context
    min_df=2,              # Ignore very rare terms
    max_df=0.95,           # Ignore terms in >95% of docs
    sublinear_tf=True      # Apply log scaling to term frequency
)
X_vec = vectorizer.fit_transform(X_raw)
X_vec = X_vec.toarray()
print(f"TF-IDF matrix shape: {X_vec.shape}")

# =========================
# Split Data (Train / Validation / Test)
# =========================
X_train, X_temp, y_train, y_temp = train_test_split(
    X_vec, y, test_size=0.2, stratify=y, random_state=SEED
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=SEED
)

print(f"\nData splits:")
print(f"  Train: {X_train.shape[0]} samples")
print(f"  Val:   {X_val.shape[0]} samples")
print(f"  Test:  {X_test.shape[0]} samples")

# =========================
# Torch Tensors & Dataloaders
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nUsing device: {device}")

X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.long)
X_val_t   = torch.tensor(X_val,   dtype=torch.float32)
y_val_t   = torch.tensor(y_val,   dtype=torch.long)
X_test_t  = torch.tensor(X_test,  dtype=torch.float32)
y_test_t  = torch.tensor(y_test,  dtype=torch.long)

from torch.utils.data import TensorDataset, DataLoader

train_ds = TensorDataset(X_train_t, y_train_t)
val_ds   = TensorDataset(X_val_t,   y_val_t)
test_ds  = TensorDataset(X_test_t,  y_test_t)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  drop_last=False)
val_loader   = DataLoader(val_ds,   batch_size=256,        shuffle=False, drop_last=False)
test_loader  = DataLoader(test_ds,  batch_size=256,        shuffle=False, drop_last=False)

# =========================
# Neural Network Architecture (Optimized)
# =========================
import torch.nn as nn

class NewsMLP(nn.Module):
    """
    Optimized 3-layer MLP with:
    - Deeper architecture (512 → 256 → 128)
    - Batch normalization + dropout
    - Skip connection from input to final hidden layer (residual-like)
    """
    def __init__(self, input_dim, num_classes):
        super().__init__()
        
        # Main pathway
        self.layer1 = nn.Sequential(
            nn.Linear(input_dim, HIDDEN_1),
            nn.BatchNorm1d(HIDDEN_1),
            nn.GELU(),  # Smoother activation than ReLU
            nn.Dropout(DROPOUT)
        )
        
        self.layer2 = nn.Sequential(
            nn.Linear(HIDDEN_1, HIDDEN_2),
            nn.BatchNorm1d(HIDDEN_2),
            nn.GELU(),
            nn.Dropout(DROPOUT)
        )
        
        self.layer3 = nn.Sequential(
            nn.Linear(HIDDEN_2, HIDDEN_3),
            nn.BatchNorm1d(HIDDEN_3),
            nn.GELU(),
            nn.Dropout(DROPOUT * 0.5)  # Less dropout before output
        )
        
        # Output layer
        self.classifier = nn.Linear(HIDDEN_3, num_classes)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return self.classifier(x)

input_dim = X_train_t.shape[1]
model = NewsMLP(input_dim=input_dim, num_classes=num_classes).to(device)
print(f"\nModel architecture:")
print(model)

total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")

# =========================
# Optimizer, Loss, Scheduler
# =========================
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)

# Cosine annealing with warm restarts for better convergence
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=10, T_mult=2, eta_min=1e-5
)

# =========================
# Early Stopping Class
# =========================
class EarlyStopping:
    """
    Early stopping based on validation accuracy (not loss).
    """
    def __init__(self, patience=5, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_model_state = None
    
    def __call__(self, val_acc, model):
        score = val_acc
        
        if self.best_score is None:
            self.best_score = score
            self.best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
            self.counter = 0
    
    def restore_best_model(self, model):
        if self.best_model_state is not None:
            model.load_state_dict(self.best_model_state)

# =========================
# Training Function
# =========================
def train(num_epochs=MAX_EPOCHS, patience=PATIENCE):
    """
    Train with early stopping based on validation accuracy.
    Uses gradient clipping for stability.
    """
    print(f"\n{'='*70}")
    print(f"Starting training (max {num_epochs} epochs, patience={patience})")
    print(f"Hyperparameters: LR={LEARNING_RATE}, WD={WEIGHT_DECAY}, Batch={BATCH_SIZE}")
    print(f"{'='*70}")
    
    early_stopping = EarlyStopping(patience=patience, min_delta=0.002)
    best_val_acc = 0.0
    
    for epoch in range(num_epochs):
        # ---- Training Phase ----
        model.train()
        running_loss, running_correct, total = 0.0, 0, 0
        
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            preds = logits.argmax(dim=1)
            running_loss += loss.item() * xb.size(0)
            running_correct += (preds == yb).sum().item()
            total += xb.size(0)
        
        train_loss = running_loss / total
        train_acc = running_correct / total
        
        # Step scheduler
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # ---- Validation Phase ----
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                loss = criterion(logits, yb)
                
                preds = logits.argmax(dim=1)
                val_loss += loss.item() * xb.size(0)
                val_correct += (preds == yb).sum().item()
                val_total += xb.size(0)
        
        val_loss = val_loss / val_total
        val_acc = val_correct / val_total
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            marker = " ★ Best"
        else:
            marker = ""
        
        print(f"Epoch {epoch+1:2d}/{num_epochs} | "
              f"Train: {train_loss:.4f} / {train_acc:.4f} | "
              f"Val: {val_loss:.4f} / {val_acc:.4f} | "
              f"LR: {current_lr:.2e}{marker}")
        
        # Check early stopping (based on accuracy)
        early_stopping(val_acc, model)
        if early_stopping.early_stop:
            print(f"\n⚡ Early stopping triggered at epoch {epoch+1}")
            break
    
    # Restore best model
    early_stopping.restore_best_model(model)
    print(f"\n✓ Restored best model (val_acc={early_stopping.best_score:.4f})")
    print(f"✓ Best validation accuracy: {best_val_acc:.4f} ({best_val_acc*100:.2f}%)")

# =========================
# Evaluation Function
# =========================
def evaluate():
    """
    Evaluate on test set with detailed metrics.
    """
    print(f"\n{'='*70}")
    print("Evaluating on test set...")
    print(f"{'='*70}")
    
    model.eval()
    all_preds, all_targets, all_probs = [], [], []
    
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)
            all_preds.append(preds.cpu().numpy())
            all_targets.append(yb.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
    
    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_targets)
    y_probs = np.concatenate(all_probs)
    
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, top_k_accuracy_score
    
    test_acc = accuracy_score(y_true, y_pred)
    top3_acc = top_k_accuracy_score(y_true, y_probs, k=3)
    top5_acc = top_k_accuracy_score(y_true, y_probs, k=5)
    
    print(f"\n🎯 Test Accuracy:     {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"🎯 Top-3 Accuracy:    {top3_acc:.4f} ({top3_acc*100:.2f}%)")
    print(f"🎯 Top-5 Accuracy:    {top5_acc:.4f} ({top5_acc*100:.2f}%)")
    
    print(f"\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=data.target_names, digits=3))
    
    print(f"\nConfusion Matrix:")
    cm = confusion_matrix(y_true, y_pred)
    print(cm)
    
    # Per-class accuracy
    print(f"\nPer-class Accuracy:")
    for i, name in enumerate(data.target_names):
        mask = y_true == i
        class_acc = (y_pred[mask] == y_true[mask]).mean()
        print(f"  {name:30s}: {class_acc:.3f}")
    
    return test_acc, y_pred, y_true

# =========================
# Main Execution
# =========================
if __name__ == "__main__":
    train(num_epochs=MAX_EPOCHS, patience=PATIENCE)
    test_acc, y_pred, y_true = evaluate()
