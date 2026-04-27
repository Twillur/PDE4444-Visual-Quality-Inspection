"""
preprocess.py
-------------
Reads raw images from data/raw/pass and data/raw/fail,
builds a metadata CSV, resizes all images to 224x224,
and splits into train / val / test sets.

Run from project root:
    py -3.11 scripts/preprocess.py
"""

import os
import shutil
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

# ── Config ────────────────────────────────────────────────────────────────────
RAW_DIR       = "data/raw"
PROCESSED_DIR = "data/processed"
METADATA_CSV  = "data/metadata.csv"
IMG_SIZE      = (224, 224)
RANDOM_SEED   = 42

# ── Step 1: Build metadata CSV from raw images ────────────────────────────────
def build_metadata():
    rows = []
    for label in ["pass", "fail"]:
        folder = os.path.join(RAW_DIR, label)
        for fname in os.listdir(folder):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                rows.append({
                    "path":  os.path.join(folder, fname),
                    "label": label
                })
    df = pd.DataFrame(rows)
    df.to_csv(METADATA_CSV, index=False)
    print(f"[preprocess] Found {len(df)} images: "
          f"{(df.label=='pass').sum()} PASS, {(df.label=='fail').sum()} FAIL")
    return df

# ── Step 2: Split into train / val / test ─────────────────────────────────────
def split_data(df):
    train_df, temp_df = train_test_split(
        df, test_size=0.30, stratify=df["label"], random_state=RANDOM_SEED
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, stratify=temp_df["label"], random_state=RANDOM_SEED
    )
    print(f"[preprocess] Split -> Train: {len(train_df)}, "
          f"Val: {len(val_df)}, Test: {len(test_df)}")
    return train_df, val_df, test_df

# ── Step 3: Copy & resize images into processed folder ───────────────────────
def save_split(df, split_name):
    for _, row in df.iterrows():
        dest_dir = os.path.join(PROCESSED_DIR, split_name, row["label"])
        os.makedirs(dest_dir, exist_ok=True)
        fname    = os.path.basename(row["path"])
        dest_path = os.path.join(dest_dir, fname)
        img = Image.open(row["path"]).convert("L").convert("RGB")  # greyscale
        img = img.resize(IMG_SIZE, Image.LANCZOS)
        img.save(dest_path)
    print(f"[preprocess] Saved '{split_name}' split ({len(df)} images)")

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if os.path.exists(PROCESSED_DIR):
        shutil.rmtree(PROCESSED_DIR)
    os.makedirs(PROCESSED_DIR)

    df = build_metadata()
    train_df, val_df, test_df = split_data(df)

    save_split(train_df, "train")
    save_split(val_df,   "val")
    save_split(test_df,  "test")

    # Save split info back to metadata CSV
    train_df = train_df.copy(); train_df["split"] = "train"
    val_df   = val_df.copy();   val_df["split"]   = "val"
    test_df  = test_df.copy();  test_df["split"]  = "test"
    pd.concat([train_df, val_df, test_df]).to_csv(METADATA_CSV, index=False)

    print("[preprocess] Done! Processed dataset ready in data/processed/")