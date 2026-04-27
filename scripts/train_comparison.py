"""
train_comparison.py
-------------------
Trains MobileNetV2 with 4 combinations of activation + optimiser:
  1. ReLU  + Adam
  2. ReLU  + SGD
  3. GELU  + Adam
  4. GELU  + SGD

Generates a single 2x2 convergence plot saved to:
  report/activation_optimizer_comparison.png

Run from project root:
    py -3.11 scripts/train_comparison.py
"""

import os, copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt

# ── Config ────────────────────────────────────────────────────────────────────
PROCESSED_DIR = "data/processed"
IMG_SIZE      = 224
BATCH_SIZE    = 16
NUM_EPOCHS    = 15          # 15 epochs is enough to show convergence trends
LR            = 0.001
RANDOM_SEED   = 42
torch.manual_seed(RANDOM_SEED)

MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.3, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])
val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

def get_loaders():
    train_ds = datasets.ImageFolder(os.path.join(PROCESSED_DIR, "train"), transform=train_transform)
    val_ds   = datasets.ImageFolder(os.path.join(PROCESSED_DIR, "val"),   transform=val_transform)
    return (DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True),
            DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False))

# ── Build MobileNetV2 with swappable activation in classifier head ─────────────
def build_model(activation='relu', num_classes=2):
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    # Freeze all base layers except last 5
    for param in model.features.parameters():
        param.requires_grad = False
    for param in model.features[-5:].parameters():
        param.requires_grad = True

    act_fn = nn.ReLU() if activation == 'relu' else nn.GELU()
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        act_fn,
        nn.Linear(model.last_channel, num_classes)
    )
    return model

# ── Single training run ────────────────────────────────────────────────────────
def train_one(activation, optimizer_name, train_loader, val_loader):
    label = f"{activation.upper()} + {optimizer_name}"
    print(f"\n[comparison] Training: {label}")

    model  = build_model(activation=activation)
    device = torch.device("cpu")
    model  = model.to(device)
    criterion = nn.CrossEntropyLoss()

    params = [
        {"params": model.features[-5:].parameters(), "lr": LR * 0.1},
        {"params": model.classifier.parameters(),    "lr": LR},
    ]
    if optimizer_name == 'Adam':
        optimizer = optim.Adam(params)
    else:  # SGD
        optimizer = optim.SGD(params, momentum=0.9, weight_decay=1e-4)

    val_accs, train_accs = [], []
    val_losses, train_losses = [], []

    for epoch in range(NUM_EPOCHS):
        # Train
        model.train()
        run_loss, correct, total = 0.0, 0, 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            out  = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            run_loss += loss.item() * imgs.size(0)
            correct  += (out.argmax(1) == labels).sum().item()
            total    += labels.size(0)
        train_losses.append(run_loss / total)
        train_accs.append(correct / total)

        # Validate
        model.eval()
        v_loss, v_correct, v_total = 0.0, 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                out    = model(imgs)
                v_loss += criterion(out, labels).item() * imgs.size(0)
                v_correct += (out.argmax(1) == labels).sum().item()
                v_total   += labels.size(0)
        val_losses.append(v_loss / v_total)
        val_accs.append(v_correct / v_total)

        print(f"  Epoch {epoch+1:02d}/{NUM_EPOCHS}  "
              f"Train Acc: {train_accs[-1]*100:.1f}%  Val Acc: {val_accs[-1]*100:.1f}%")

    print(f"  >> Best Val Acc: {max(val_accs)*100:.1f}%")
    return train_losses, val_losses, train_accs, val_accs

# ── Plot all 4 combinations ────────────────────────────────────────────────────
def plot_comparison(results):
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    fig.suptitle("Activation Function & Optimiser Comparison — Convergence Plots",
                 fontsize=13, fontweight='bold')

    colors = {'Train': '#2196F3', 'Val': '#FF5722'}
    epochs = range(1, NUM_EPOCHS + 1)

    combos = [
        ("ReLU + Adam",  "relu",  "Adam"),
        ("ReLU + SGD",   "relu",  "SGD"),
        ("GELU + Adam",  "gelu",  "Adam"),
        ("GELU + SGD",   "gelu",  "SGD"),
    ]

    for col, (label, act, opt) in enumerate(combos):
        tr_loss, v_loss, tr_acc, v_acc = results[(act, opt)]

        # Loss plot (top row)
        ax_loss = axes[0][col]
        ax_loss.plot(epochs, tr_loss, color=colors['Train'], label='Train Loss', linewidth=1.8)
        ax_loss.plot(epochs, v_loss,  color=colors['Val'],   label='Val Loss',   linewidth=1.8, linestyle='--')
        ax_loss.set_title(label, fontweight='bold', fontsize=10)
        ax_loss.set_xlabel("Epoch", fontsize=8)
        ax_loss.set_ylabel("Loss", fontsize=8)
        ax_loss.legend(fontsize=7)
        ax_loss.grid(alpha=0.3)

        # Accuracy plot (bottom row)
        ax_acc = axes[1][col]
        ax_acc.plot(epochs, [a*100 for a in tr_acc], color=colors['Train'], label='Train Acc', linewidth=1.8)
        ax_acc.plot(epochs, [a*100 for a in v_acc],  color=colors['Val'],   label='Val Acc',   linewidth=1.8, linestyle='--')
        ax_acc.set_xlabel("Epoch", fontsize=8)
        ax_acc.set_ylabel("Accuracy (%)", fontsize=8)
        ax_acc.set_ylim(40, 105)
        ax_acc.legend(fontsize=7)
        ax_acc.grid(alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join("report", "activation_optimizer_comparison.png")
    os.makedirs("report", exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"\n[comparison] Plot saved to {out_path}")

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    train_loader, val_loader = get_loaders()

    results = {}
    for activation in ['relu', 'gelu']:
        for optimizer_name in ['Adam', 'SGD']:
            tr_loss, v_loss, tr_acc, v_acc = train_one(
                activation, optimizer_name, train_loader, val_loader
            )
            results[(activation, optimizer_name)] = (tr_loss, v_loss, tr_acc, v_acc)

    plot_comparison(results)
    print("\n[comparison] Done. Upload report/activation_optimizer_comparison.png to Overleaf.")
