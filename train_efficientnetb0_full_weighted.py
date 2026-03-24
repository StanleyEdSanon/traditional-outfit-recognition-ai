# train_efficientnetb0_full_weighted_sqrt.py
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as T

from dataset_loader import MultiOutputOutfitDataset
from models.multioutput_efficientnet import MultiOutputEfficientNetB0

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def compute_class_weights(
    csv_path: str,
    col: str,
    num_classes: int,
    mode: str = "sqrt",   # "inv" or "sqrt"
    normalize: bool = True,
):
    """
    Returns torch.FloatTensor [num_classes] with class weights computed from TRAIN CSV ONLY.

    mode:
      - "inv"  : 1 / freq
      - "sqrt" : 1 / sqrt(freq)  (recommended: less aggressive, more stable)
    """
    df = pd.read_csv(csv_path)
    counts = df[col].value_counts().to_dict()

    freq = np.array([counts.get(i, 0) for i in range(num_classes)], dtype=np.float64)
    freq = np.maximum(freq, 1.0)  # avoid division by zero

    if mode == "inv":
        w = 1.0 / freq
    elif mode == "sqrt":
        w = 1.0 / np.sqrt(freq)
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'inv' or 'sqrt'.")

    if normalize:
        w = w / w.mean()  # keep average weight ~1.0 for stable loss scale

    return torch.tensor(w, dtype=torch.float32)


def build_transforms(aug: bool):
    if aug:
        train_tf = T.Compose([
            T.RandomResizedCrop(224, scale=(0.7, 1.0)),
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])
    else:
        train_tf = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])

    val_tf = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])
    return train_tf, val_tf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", required=True)
    ap.add_argument("--val_csv", required=True)
    ap.add_argument("--save", default="efficientnetb0_full_best.pt")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--num_workers", type=int, default=4)

    # weighting toggles
    ap.add_argument("--w_name", action="store_true", help="Enable class weights for name head")
    ap.add_argument("--w_type", action="store_true", help="Enable class weights for type head")
    ap.add_argument("--w_sleeve", action="store_true", help="Enable class weights for sleeve head")

    # class weight mode
    ap.add_argument("--w_mode", default="sqrt", choices=["sqrt", "inv"],
                    help="Class-weighting mode: sqrt (recommended) or inv (aggressive)")

    # head loss multipliers (lets you downweight unstable heads like sleeve)
    ap.add_argument("--lambda_name", type=float, default=1.0)
    ap.add_argument("--lambda_type", type=float, default=1.0)
    ap.add_argument("--lambda_sleeve", type=float, default=1.0)

    # augmentation
    ap.add_argument("--aug", action="store_true", help="Enable stronger train augmentation")

    # reproducibility
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # ---------------- SEED ----------------
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    print("Using device:", DEVICE)
    print(f"Weight mode: {args.w_mode}")
    print(f"Lambdas: name={args.lambda_name}, type={args.lambda_type}, sleeve={args.lambda_sleeve}")

    train_tf, val_tf = build_transforms(args.aug)

    # FULL images => require_roi=False
    train_ds = MultiOutputOutfitDataset(args.train_csv, transform=train_tf)  # no require_roi arg
    val_ds   = MultiOutputOutfitDataset(args.val_csv,   transform=val_tf)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch, shuffle=True,
        num_workers=args.num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )

    model = MultiOutputEfficientNetB0().to(DEVICE)

    # ---------------- CLASS WEIGHTS (TRAIN ONLY) ----------------
    w_name = compute_class_weights(args.train_csv, "name_idx",   3, mode=args.w_mode) if args.w_name else None
    w_type = compute_class_weights(args.train_csv, "type_idx",   3, mode=args.w_mode) if args.w_type else None
    w_slv  = compute_class_weights(args.train_csv, "sleeve_idx", 4, mode=args.w_mode) if args.w_sleeve else None

    if w_name is not None: print("Name weights  :", w_name.numpy())
    if w_type is not None: print("Type weights  :", w_type.numpy())
    if w_slv  is not None: print("Sleeve weights:", w_slv.numpy())

    crit_name = nn.CrossEntropyLoss(weight=w_name.to(DEVICE) if w_name is not None else None)
    crit_type = nn.CrossEntropyLoss(weight=w_type.to(DEVICE) if w_type is not None else None)
    crit_slv  = nn.CrossEntropyLoss(weight=w_slv.to(DEVICE)  if w_slv  is not None else None)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val = float("inf")

    for epoch in range(1, args.epochs + 1):
        # ---------------- TRAIN ----------------
        model.train()
        tr_loss = 0.0

        for imgs, (y_name, y_type, y_sleeve) in train_loader:
            imgs = imgs.to(DEVICE, non_blocking=True)
            y_name = y_name.to(DEVICE, non_blocking=True)
            y_type = y_type.to(DEVICE, non_blocking=True)
            y_sleeve = y_sleeve.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            p_name, p_type, p_sleeve = model(imgs)

            loss = (
                args.lambda_name   * crit_name(p_name, y_name) +
                args.lambda_type   * crit_type(p_type, y_type) +
                args.lambda_sleeve * crit_slv(p_sleeve, y_sleeve)
            )

            loss.backward()
            optimizer.step()

            tr_loss += loss.item() * imgs.size(0)

        tr_loss /= len(train_ds)

        # ---------------- VAL ----------------
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
                    args.lambda_name   * crit_name(p_name, y_name) +
                    args.lambda_type   * crit_type(p_type, y_type) +
                    args.lambda_sleeve * crit_slv(p_sleeve, y_sleeve)
                )
                val_loss += loss.item() * imgs.size(0)

        val_loss /= len(val_ds)

        print(f"Epoch [{epoch:02d}/{args.epochs}] | Train Loss: {tr_loss:.4f} | Val Loss: {val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), args.save)
            print("🔥 Saved best:", args.save)

    print("🎉 Training complete. Best val loss:", best_val)


if __name__ == "__main__":
    main()