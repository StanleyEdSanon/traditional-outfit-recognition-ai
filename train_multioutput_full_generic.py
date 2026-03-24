import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as T
import pandas as pd

from dataset_loader import MultiOutputOutfitDataset
from models.multioutput_resnet import MultiOutputResNet18
from models.multioutput_efficientnet import MultiOutputEfficientNetB0
from models.multioutput_deit import MultiOutputDeiT

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 224


# ---------------- CSV AUDIT ----------------
def audit_csv(csv_path: str, mode: str):
    """
    mode:
      - "full" : expects NON-roi paths
      - "roi"  : expects roi_crops paths
    """
    df = pd.read_csv(csv_path)

    required = {"image", "name_idx", "type_idx", "sleeve_idx"}
    if not required.issubset(df.columns):
        raise SystemExit(f"[FATAL] CSV {csv_path} missing columns. Found: {list(df.columns)}")

    df["image"] = df["image"].astype(str)

    print(f"\nCSV: {csv_path}")
    print("Sample paths:")
    for p in df["image"].head(5).tolist():
        print(" ", p)

    if mode == "full":
        bad = df[df["image"].str.contains(r"(^|/|\\)roi_crops($|/|\\)", regex=True, case=False)]
        if len(bad) > 0:
            raise SystemExit(f"[FATAL] FULL mode CSV contains ROI path. Example: {bad['image'].iloc[0]}")
    elif mode == "roi":
        bad = df[~df["image"].str.contains(r"(^|/|\\)roi_crops($|/|\\)", regex=True, case=False)]
        if len(bad) > 0:
            raise SystemExit(f"[FATAL] ROI mode CSV contains non-ROI path. Example: {bad['image'].iloc[0]}")
    else:
        raise SystemExit("[FATAL] mode must be 'full' or 'roi'")

    missing = df[~df["image"].apply(os.path.exists)]
    if len(missing) > 0:
        ex = missing["image"].iloc[0]
        raise SystemExit(f"[FATAL] Missing file on disk. Example: {ex}")

    print("\nDistributions:")
    print(" name_idx  :", df["name_idx"].value_counts().sort_index().to_dict())
    print(" type_idx  :", df["type_idx"].value_counts().sort_index().to_dict())
    print(" sleeve_idx:", df["sleeve_idx"].value_counts().sort_index().to_dict())

    aug_count = int(df["image"].str.contains(r"_aug\d+\.", regex=True, case=False).sum())
    print(" aug_count :", aug_count, "/", len(df))

    print("✅ CSV audit passed")


# ---------------- MODEL FACTORY ----------------
def get_model(name: str):
    name = name.strip().lower()
    if name == "resnet":
        return MultiOutputResNet18()
    if name == "efficientnet":
        return MultiOutputEfficientNetB0()
    if name == "deit":
        return MultiOutputDeiT()
    raise ValueError(f"Unknown model: {name}")


# ---------------- TRAIN ----------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["resnet", "efficientnet", "deit"])
    parser.add_argument("--train_csv", required=True)
    parser.add_argument("--val_csv", required=True)
    parser.add_argument("--save", required=True)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--mode", choices=["full", "roi"], default="full")
    args = parser.parse_args()

    print("Using device:", DEVICE)
    print("Model:", args.model)

    audit_csv(args.train_csv, mode=args.mode)
    audit_csv(args.val_csv, mode=args.mode)

    transform = T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])

    train_ds = MultiOutputOutfitDataset(args.train_csv, transform=transform)
    val_ds   = MultiOutputOutfitDataset(args.val_csv, transform=transform)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch, shuffle=True,
        num_workers=args.num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )

    model = get_model(args.model).to(DEVICE)

    criterion_name   = nn.CrossEntropyLoss()
    criterion_type   = nn.CrossEntropyLoss()
    criterion_sleeve = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0

        for imgs, (y_name, y_type, y_sleeve) in train_loader:
            imgs = imgs.to(DEVICE, non_blocking=True)
            y_name = y_name.to(DEVICE, non_blocking=True)
            y_type = y_type.to(DEVICE, non_blocking=True)
            y_sleeve = y_sleeve.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            p_name, p_type, p_sleeve = model(imgs)
            loss = (
                criterion_name(p_name, y_name)
                + criterion_type(p_type, y_type)
                + criterion_sleeve(p_sleeve, y_sleeve)
            )

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * imgs.size(0)

        train_loss /= len(train_ds)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, (y_name, y_type, y_sleeve) in val_loader:
                imgs = imgs.to(DEVICE, non_blocking=True)
                y_name = y_name.to(DEVICE, non_blocking=True)
                y_type = y_type.to(DEVICE, non_blocking=True)
                y_sleeve = y_sleeve.to(DEVICE, non_blocking=True)

                p_name, p_type, p_sleeve = model(imgs)
                loss = (
                    criterion_name(p_name, y_name)
                    + criterion_type(p_type, y_type)
                    + criterion_sleeve(p_sleeve, y_sleeve)
                )
                val_loss += loss.item() * imgs.size(0)

        val_loss /= len(val_ds)

        print(f"Epoch [{epoch:02d}/{args.epochs}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), args.save)
            print("🔥 Saved best:", args.save)

    print("\n🎉 Training complete.")
    print("Best val loss:", best_val)


if __name__ == "__main__":
    main()