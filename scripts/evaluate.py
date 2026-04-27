"""
evaluate.py
-----------
Loads the trained model and evaluates it on the test set.
Outputs: accuracy, precision, recall, F1, confusion matrix,
         and a PCA visualization of learned features.

Run from project root:
    py -3.11 scripts/evaluate.py
"""

import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)
from sklearn.decomposition import PCA

# ── Config ────────────────────────────────────────────────────────────────────
PROCESSED_DIR = "data/processed"
MODEL_PATH    = "models/best_model.pth"
IMG_SIZE      = 224
BATCH_SIZE    = 16
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

# ── Transform (no augmentation for evaluation) ────────────────────────────────
eval_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

# ── Load model ────────────────────────────────────────────────────────────────
def load_model(num_classes=2):
    model = models.mobilenet_v2(weights=None)
    model.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(model.last_channel, num_classes)
    )
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    return model

# ── Run inference on test set ─────────────────────────────────────────────────
def run_inference(model, test_loader):
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for imgs, labels in test_loader:
            outputs = model(imgs)
            probs   = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.numpy())
    return np.array(all_preds), np.array(all_labels), np.array(all_probs)

# ── Print metrics ─────────────────────────────────────────────────────────────
def print_metrics(preds, labels, classes):
    print("\n[evaluate] -- Test Set Results --------------------------")
    print(f"  Accuracy  : {accuracy_score(labels, preds)*100:.1f}%")
    print(f"  Precision : {precision_score(labels, preds, average='binary')*100:.1f}%")
    print(f"  Recall    : {recall_score(labels, preds, average='binary')*100:.1f}%")
    print(f"  F1 Score  : {f1_score(labels, preds, average='binary')*100:.1f}%")
    print("\n" + classification_report(labels, preds, target_names=classes))

# ── Confusion matrix plot ─────────────────────────────────────────────────────
def plot_confusion_matrix(preds, labels, classes):
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=classes, yticklabels=classes)
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig("models/confusion_matrix.png", dpi=150)
    print("[evaluate] Confusion matrix saved to models/confusion_matrix.png")

# ── PCA feature visualization ─────────────────────────────────────────────────
def plot_pca_features(model, test_loader, classes):
    """
    Extract features from the second-to-last layer and visualise
    them in 2D using PCA. Shows how well the model separates the two classes.
    Connects to the PCA course topic.
    """
    features, labels_list = [], []

    # Hook to capture feature map output before classifier
    activation = {}
    def hook_fn(module, input, output):
        activation["features"] = output.detach()

    # Register hook on the adaptive pool layer
    handle = model.features[-1].register_forward_hook(hook_fn)

    with torch.no_grad():
        for imgs, labels in test_loader:
            _ = model(imgs)
            feat = activation["features"]
            feat = torch.nn.functional.adaptive_avg_pool2d(feat, 1)
            feat = feat.view(feat.size(0), -1)
            features.append(feat.numpy())
            labels_list.extend(labels.numpy())

    handle.remove()
    features = np.vstack(features)
    labels_arr = np.array(labels_list)

    # PCA to 2 components
    pca  = PCA(n_components=2, random_state=42)
    proj = pca.fit_transform(features)

    plt.figure(figsize=(6, 5))
    colors = ["#2ecc71", "#e74c3c"]
    for i, cls in enumerate(classes):
        mask = labels_arr == i
        plt.scatter(proj[mask, 0], proj[mask, 1],
                    c=colors[i], label=cls, alpha=0.7, edgecolors="k", linewidths=0.3)
    plt.title("PCA of CNN Features (Test Set)")
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("models/pca_features.png", dpi=150)
    print("[evaluate] PCA feature plot saved to models/pca_features.png")

# ── Failure case analysis ──────────────────────────────────────────────────────
def show_failure_cases(model, test_loader, classes):
    """Save a grid of misclassified images for failure analysis in the report."""
    failures = []
    model.eval()
    with torch.no_grad():
        for imgs, labels in test_loader:
            outputs = model(imgs)
            _, preds = torch.max(outputs, 1)
            for i in range(len(labels)):
                if preds[i] != labels[i]:
                    failures.append((imgs[i], labels[i].item(), preds[i].item()))
            if len(failures) >= 8:
                break

    if not failures:
        print("[evaluate] No failure cases found on test set!")
        return

    n = min(len(failures), 8)
    fig, axes = plt.subplots(1, n, figsize=(3*n, 3))
    axes = axes.flatten() if n > 1 else [axes]

    MEAN_t = torch.tensor(MEAN).view(3,1,1)
    STD_t  = torch.tensor(STD).view(3,1,1)

    for idx in range(n):
        img, true_lbl, pred_lbl = failures[idx]
        img_disp = img * STD_t + MEAN_t
        img_disp = img_disp.permute(1, 2, 0).clamp(0, 1).numpy()
        axes[idx].imshow(img_disp)
        axes[idx].set_title(
            f"True: {classes[true_lbl]}\nPred: {classes[pred_lbl]}",
            color="red", fontsize=8
        )
        axes[idx].axis("off")

    plt.suptitle("Failure Cases", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig("models/failure_cases.png", dpi=150)
    print(f"[evaluate] {n} failure cases saved to models/failure_cases.png")

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    test_ds = datasets.ImageFolder(
        os.path.join(PROCESSED_DIR, "test"), transform=eval_transform
    )
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
    classes     = test_ds.classes
    print(f"[evaluate] Classes: {classes}, Test samples: {len(test_ds)}")

    model = load_model(num_classes=len(classes))
    preds, labels, probs = run_inference(model, test_loader)

    print_metrics(preds, labels, classes)
    plot_confusion_matrix(preds, labels, classes)
    plot_pca_features(model, test_loader, classes)
    show_failure_cases(model, test_loader, classes)

    print("\n[evaluate] All outputs saved to models/")
