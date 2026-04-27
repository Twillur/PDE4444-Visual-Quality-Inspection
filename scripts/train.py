"""
train.py
--------
1. Trains a Logistic Regression baseline on PCA-reduced image features.
2. Fine-tunes a MobileNetV2 CNN (transfer learning) for PASS/FAIL classification.
Saves the best model to models/best_model.pth

Run from project root:
    py -3.11 scripts/train.py
"""

import os
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt

# ── Config ────────────────────────────────────────────────────────────────────
PROCESSED_DIR = "data/processed"
MODEL_PATH    = "models/best_model.pth"
IMG_SIZE      = 224
BATCH_SIZE    = 16
NUM_EPOCHS    = 30
LR            = 0.001
RANDOM_SEED   = 42
torch.manual_seed(RANDOM_SEED)

# ── Transforms ────────────────────────────────────────────────────────────────
# ImageNet normalisation stats (required for MobileNetV2 pretrained weights)
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

val_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

# ── Data Loaders ──────────────────────────────────────────────────────────────
def get_loaders():
    train_ds = datasets.ImageFolder(
        os.path.join(PROCESSED_DIR, "train"), transform=train_transform
    )
    val_ds = datasets.ImageFolder(
        os.path.join(PROCESSED_DIR, "val"), transform=val_transform
    )
    test_ds = datasets.ImageFolder(
        os.path.join(PROCESSED_DIR, "test"), transform=val_transform
    )
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False)
    print(f"[train] Classes: {train_ds.classes}")
    print(f"[train] Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")
    return train_loader, val_loader, test_loader, train_ds.classes

# ── Baseline: Logistic Regression on PCA features ────────────────────────────
def run_baseline(train_loader, test_loader):
    print("\n[train] --- Logistic Regression Baseline ---")

    def extract(loader):
        feats, labels = [], []
        for imgs, lbls in loader:
            flat = imgs.view(imgs.size(0), -1).numpy()
            feats.append(flat)
            labels.append(lbls.numpy())
        return np.vstack(feats), np.concatenate(labels)

    X_train, y_train = extract(train_loader)
    X_test,  y_test  = extract(test_loader)

    # PCA: reduce to 50 components (connects to PCA course topic)
    pca = PCA(n_components=50, random_state=RANDOM_SEED)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca  = pca.transform(X_test)

    clf = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)
    clf.fit(X_train_pca, y_train)
    acc = accuracy_score(y_test, clf.predict(X_test_pca))
    print(f"[train] Baseline Logistic Regression accuracy: {acc*100:.1f}%")
    return acc

# ── Custom CNN (designed from scratch) ───────────────────────────────────────
class CustomCNN(nn.Module):
    """
    Fully custom CNN designed for PASS/FAIL tissue box classification.
    Architecture:
      Input  : 224x224x3 RGB image
      Block 1: Conv(3->32)   -> BN -> ReLU -> MaxPool  => 112x112x32
      Block 2: Conv(32->64)  -> BN -> ReLU -> MaxPool  =>  56x56x64
      Block 3: Conv(64->128) -> BN -> ReLU -> MaxPool  =>  28x28x128
      Block 4: Conv(128->256)-> BN -> ReLU -> MaxPool  =>  14x14x256
      GAP    : AdaptiveAvgPool => 1x1x256
      Head   : Linear(256->128) -> ReLU -> Dropout -> Linear(128->2)
      Output : Softmax (2 classes: FAIL, PASS)
    """
    def __init__(self, num_classes=2):
        super(CustomCNN, self).__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # Block 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def train_custom_cnn(train_loader, val_loader, num_classes=2):
    print("\n[train] --- Custom CNN Training (from scratch) ---")
    device = torch.device("cpu")
    model  = CustomCNN(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=4
    )

    best_acc, best_weights = 0.0, copy.deepcopy(model.state_dict())
    train_losses, val_losses, train_accs, val_accs = [], [], [], []

    for epoch in range(NUM_EPOCHS):
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

        model.eval()
        v_loss, v_correct, v_total = 0.0, 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                out     = model(imgs)
                v_loss += criterion(out, labels).item() * imgs.size(0)
                v_correct += (out.argmax(1) == labels).sum().item()
                v_total   += labels.size(0)
        val_losses.append(v_loss / v_total)
        val_accs.append(v_correct / v_total)
        scheduler.step(val_accs[-1])

        print(f"  Epoch [{epoch+1:02d}/{NUM_EPOCHS}] "
              f"Train Acc: {train_accs[-1]*100:.1f}%  Val Acc: {val_accs[-1]*100:.1f}%")

        if val_accs[-1] > best_acc:
            best_acc     = val_accs[-1]
            best_weights = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_weights)
    print(f"\n[train] Custom CNN Best Val Accuracy: {best_acc*100:.1f}%")
    torch.save(model.state_dict(), "models/custom_cnn.pth")
    print("[train] Custom CNN saved to models/custom_cnn.pth")
    return model, train_losses, val_losses, train_accs, val_accs


# ── CNN: MobileNetV2 Transfer Learning ───────────────────────────────────────
def build_model(num_classes=2):
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    for param in model.features.parameters():
        param.requires_grad = False
    for param in model.features[-5:].parameters():
        param.requires_grad = True
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(model.last_channel, num_classes)
    )
    return model

def train_model(model, train_loader, val_loader):
    device    = torch.device("cpu")
    model     = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam([
        {"params": model.features[-5:].parameters(), "lr": LR * 0.1},
        {"params": model.classifier.parameters(),    "lr": LR},
    ])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=4
    )

    best_acc    = 0.0
    best_weights = copy.deepcopy(model.state_dict())
    train_losses, val_losses = [], []
    train_accs,   val_accs   = [], []

    print("\n[train] --- MobileNetV2 CNN Training ---")
    for epoch in range(NUM_EPOCHS):
        # ── Training phase ─────────────────────────────────────────────────
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss    = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)
            _, preds = torch.max(outputs, 1)
            correct  += (preds == labels).sum().item()
            total    += labels.size(0)

        epoch_loss = running_loss / total
        epoch_acc  = correct / total
        train_losses.append(epoch_loss)
        train_accs.append(epoch_acc)

        # ── Validation phase ───────────────────────────────────────────────
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs  = model(imgs)
                loss     = criterion(outputs, labels)
                val_loss += loss.item() * imgs.size(0)
                _, preds  = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total   += labels.size(0)

        v_loss = val_loss / val_total
        v_acc  = val_correct / val_total
        val_losses.append(v_loss)
        val_accs.append(v_acc)
        scheduler.step(v_acc)

        print(f"  Epoch [{epoch+1:02d}/{NUM_EPOCHS}] "
              f"Train Loss: {epoch_loss:.4f}  Train Acc: {epoch_acc*100:.1f}%  |  "
              f"Val Loss: {v_loss:.4f}  Val Acc: {v_acc*100:.1f}%")

        if v_acc > best_acc:
            best_acc     = v_acc
            best_weights = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_weights)
    print(f"\n[train] Best Val Accuracy: {best_acc*100:.1f}%")
    return model, train_losses, val_losses, train_accs, val_accs

# ── Plot training curves ──────────────────────────────────────────────────────
def plot_curves(train_losses, val_losses, train_accs, val_accs):
    os.makedirs("models", exist_ok=True)
    epochs = range(1, len(train_losses) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(epochs, train_losses, label="Train Loss")
    ax1.plot(epochs, val_losses,   label="Val Loss")
    ax1.set_title("Loss per Epoch"); ax1.set_xlabel("Epoch")
    ax1.legend()

    ax2.plot(epochs, train_accs, label="Train Acc")
    ax2.plot(epochs, val_accs,   label="Val Acc")
    ax2.set_title("Accuracy per Epoch"); ax2.set_xlabel("Epoch")
    ax2.legend()

    plt.tight_layout()
    plt.savefig("models/training_curves.png", dpi=150)
    print("[train] Training curves saved to models/training_curves.png")

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    train_loader, val_loader, test_loader, classes = get_loaders()

    # Step 1: Baseline
    run_baseline(train_loader, test_loader)

    # Step 2: Custom CNN (from scratch)
    os.makedirs("models", exist_ok=True)
    custom_model, ctr_loss, cv_loss, ctr_acc, cv_acc = train_custom_cnn(
        train_loader, val_loader, num_classes=len(classes)
    )
    plot_curves(ctr_loss, cv_loss, ctr_acc, cv_acc)
    import shutil
    shutil.copy("models/training_curves.png", "models/custom_cnn_curves.png")

    # Step 3: MobileNetV2 (transfer learning)
    model = build_model(num_classes=len(classes))
    model, tr_loss, v_loss, tr_acc, v_acc = train_model(
        model, train_loader, val_loader
    )

    # Save best model
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"[train] MobileNetV2 saved to {MODEL_PATH}")

    # Plot curves
    plot_curves(tr_loss, v_loss, tr_acc, v_acc)
