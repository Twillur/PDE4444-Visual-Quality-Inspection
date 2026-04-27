# Tissue Box Visual Quality Inspection

**PDE4444 — Machine Learning for Engineers**  
Middlesex University Dubai · William Kojumian · M01094013

---

A complete end-to-end machine learning pipeline that classifies tissue boxes as **PASS** (undamaged) or **FAIL** (defective) using computer vision — simulating an automated industrial quality inspection system.

---

## Results at a Glance

| Model | Test Accuracy | Notes |
|---|---|---|
| Logistic Regression (PCA baseline) | ~72% | 50 PCA components on flat pixel features |
| Custom CNN (from scratch) | ~88% | 4-block convolutional network |
| **MobileNetV2 (transfer learning)** | **~95%** | Fine-tuned on last 5 feature blocks |

---

## Project Structure

```
tissue_inspection/
├── data/
│   ├── metadata.csv          ← image paths, labels, and train/val/test splits
│   ├── samples/              ← representative pass & fail example images
│   ├── raw/                  ← full raw dataset (excluded — 2.5 GB)
│   └── processed/            ← resized/split images (auto-generated, excluded)
├── models/
│   ├── best_model.pth        ← trained MobileNetV2 weights
│   ├── custom_cnn.pth        ← trained custom CNN weights
│   ├── training_curves.png   ← loss & accuracy over epochs (MobileNetV2)
│   ├── custom_cnn_curves.png ← loss & accuracy over epochs (Custom CNN)
│   ├── confusion_matrix.png  ← test set confusion matrix
│   ├── pca_features.png      ← PCA of learned CNN features
│   └── failure_cases.png     ← misclassified examples
├── scripts/
│   ├── preprocess.py         ← resize, split, build metadata CSV
│   ├── train.py              ← baseline + custom CNN + MobileNetV2 training
│   ├── train_comparison.py   ← activation × optimiser ablation study
│   ├── evaluate.py           ← metrics, confusion matrix, PCA, failure analysis
│   ├── demo.py               ← live webcam PASS/FAIL overlay
│   └── video_test.py         ← run inference on a recorded video file
├── report/
│   ├── main.tex                       ← LaTeX report source (v1)
│   └── overleafdocfortissue_v2.txt    ← Overleaf source (v2, submitted)
└── README.md
```

---

## Dataset

Images collected manually with an iPhone under varied real-world conditions.

| Split | PASS | FAIL | Total |
|---|---|---|---|
| Train (70%) | 219 | 357 | 576 |
| Val (15%) | 47 | 77 | 124 |
| Test (15%) | 47 | 77 | 124 |
| **Total** | **313** | **511** | **824** |

**FAIL examples:** crushed corners, torn packaging, water damage, missing labels, dented boxes.  
**Capture conditions:** varied lighting (bright, dim, mixed), multiple angles (front, top, sides, diagonal), plain and cluttered backgrounds.

> The full raw dataset (~2.5 GB) is excluded from this repository. The `data/samples/` folder contains representative examples from each class.  
> `data/metadata.csv` documents every image's path, label, and split assignment.

---

## Models & Architecture

### 1 · Logistic Regression Baseline
- Flattens each 224×224 image to a 150,528-dimensional pixel vector
- Applies PCA (50 components) for dimensionality reduction
- Trains a standard `sklearn` Logistic Regression classifier
- Provides a performance floor for CNN comparison

### 2 · Custom CNN (from scratch)

```
Input: 224×224×3
  Block 1: Conv(3→32)   + BN + ReLU + MaxPool  →  112×112×32
  Block 2: Conv(32→64)  + BN + ReLU + MaxPool  →   56×56×64
  Block 3: Conv(64→128) + BN + ReLU + MaxPool  →   28×28×128
  Block 4: Conv(128→256)+ BN + ReLU + GAP      →    1×1×256
  Head:    Linear(256→128) → ReLU → Dropout(0.5) → Linear(128→2)
  Output:  Softmax → {FAIL, PASS}
```

### 3 · MobileNetV2 (Transfer Learning)
- Pretrained on ImageNet; last 5 feature blocks unfrozen for fine-tuning
- Custom classifier head: `Dropout(0.3) → Linear(1280→2)`
- Differential learning rates: `1e-4` (feature blocks) vs `1e-3` (head)
- LR scheduler: `ReduceLROnPlateau(patience=4, factor=0.5)`

### 4 · Ablation Study — Activation × Optimiser (`train_comparison.py`)

Four MobileNetV2 configurations trained over 15 epochs:

| Activation | Optimiser | Best Val Acc |
|---|---|---|
| ReLU | Adam | ~93% |
| ReLU | SGD | ~89% |
| GELU | Adam | ~94% |
| GELU | SGD | ~88% |

---

## Training Details

| Hyperparameter | Value |
|---|---|
| Image size | 224 × 224 |
| Batch size | 16 |
| Epochs | 30 (main) · 15 (ablation) |
| Optimiser | Adam |
| Base learning rate | 0.001 |
| LR schedule | ReduceLROnPlateau |
| Loss function | CrossEntropyLoss |
| Augmentation | H-flip, V-flip, rotation ±20°, perspective distortion |
| Normalisation | ImageNet mean/std `[0.485, 0.456, 0.406]` / `[0.229, 0.224, 0.225]` |
| Hardware | CPU |

---

## Course Topics Covered

| Topic | Where Used |
|---|---|
| Linear Classification | Logistic Regression baseline — `train.py` |
| PCA | Feature reduction (baseline) + feature visualisation — `evaluate.py` |
| CNN Architecture | Custom 4-block CNN designed from scratch — `train.py` |
| Transfer Learning | MobileNetV2 fine-tuning — `train.py` |
| Optimisation | Adam, SGD, learning rate scheduling — `train.py`, `train_comparison.py` |
| Activation Functions | ReLU vs GELU ablation — `train_comparison.py` |
| Evaluation Metrics | Accuracy, Precision, Recall, F1, Confusion Matrix — `evaluate.py` |
| Bias & Variance | Training/validation loss curve analysis |
| Data Preprocessing | Resize, grayscale conversion, stratified split — `preprocess.py` |
| Pandas | Metadata CSV management — `preprocess.py` |

---

## Setup & Usage

### Install dependencies

```bash
pip install torch torchvision scikit-learn matplotlib seaborn pandas pillow opencv-python
```

> Python 3.11 recommended. Tested on PyTorch 2.10.0 (CPU).

### Run the pipeline

**1. Place raw images**
```
data/raw/pass/   ← undamaged tissue boxes
data/raw/fail/   ← defective tissue boxes
```

**2. Preprocess**
```bash
py -3.11 scripts/preprocess.py
```
Resizes all images to 224×224, converts to grayscale-RGB, stratified-splits into train/val/test, writes `data/metadata.csv`.

**3. Train**
```bash
py -3.11 scripts/train.py
```
Runs three models: LR baseline → Custom CNN → MobileNetV2. Saves `models/best_model.pth`.

**4. Ablation study** *(optional)*
```bash
py -3.11 scripts/train_comparison.py
```
Trains 4 activation × optimiser combinations, saves convergence plot to `report/`.

**5. Evaluate**
```bash
py -3.11 scripts/evaluate.py
```
Prints accuracy, precision, recall, F1. Saves confusion matrix, PCA feature plot, and failure case grid to `models/`.

**6. Live demo**
```bash
py -3.11 scripts/demo.py
```
Opens webcam with real-time PASS/FAIL overlay and confidence score.  
Press **Q** to quit · Press **S** to save a frame.

> For phone camera: install **DroidCam**, connect via USB, set `CAMERA_INDEX = 1` in `demo.py`.

---

## Output Plots

| File | Description |
|---|---|
| `models/training_curves.png` | MobileNetV2 loss & accuracy per epoch |
| `models/custom_cnn_curves.png` | Custom CNN loss & accuracy per epoch |
| `models/confusion_matrix.png` | Test set confusion matrix |
| `models/pca_features.png` | 2D PCA projection of learned CNN features |
| `models/failure_cases.png` | Misclassified test image grid |
| `report/activation_optimizer_comparison.png` | Ablation study convergence plots |

---

*PDE4444 — Machine Learning for Engineers · Middlesex University Dubai · 2024–25*
