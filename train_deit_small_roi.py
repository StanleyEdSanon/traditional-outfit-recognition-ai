import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms as T

import timm
from dataset_loader import MultiOutputOutfitDataset

class MultiOutputDeiTSmall(nn.Module):
    def __init__(self, num_names=3, num_types=3, num_sleeves=4, pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(
            "deit_small_patch16_224",
            pretrained=pretrained,
            num_classes=0  # feature extractor
        )
        in_features = self.backbone.num_features  # 384 for DeiT-small

        self.head_name = nn.Linear(in_features, num_names)
        self.head_type = nn.Linear(in_features, num_types)
        self.head_sleeve = nn.Linear(in_features, num_sleeves)

    def forward(self, x):
        feat = self.backbone(x)  # [B, 384]
        return self.head_name(feat), self.head_type(feat), self.head_sleeve(feat)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", default="train_roi_aug_multioutput.csv")
    ap.add_argument("--val_csv",   default="val_roi_multioutput.csv")
    ap.add_argument("--save",      default="deit_small_roi_best.pt")
    ap.add_argument("--epochs",    type=int, default=50)
    ap.add_argument("--batch",     type=int, default=32)
    ap.add_argument("--lr",        type=float, default=1e-4)
    ap.add_argument("--workers",   type=int, default=4)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])

    train_ds = MultiOutputOutfitDataset(args.train_csv, transform=transform, require_roi=True)
    val_ds   = MultiOutputOutfitDataset(args.val_csv,   transform=transform, require_roi=True)

    train_ld = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                          num_workers=args.workers, pin_memory=True)
    val_ld   = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                          num_workers=args.workers, pin_memory=True)

    model = MultiOutputDeiTSmall(pretrained=True).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-7, verbose=True
    )

    def loss_fn(preds, labels):
        pn, pt, ps = preds
        yn, yt, ys = labels
        return (criterion(pn, yn) + criterion(pt, yt) + criterion(ps, ys))

    best_val = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        tr_loss = 0.0
        for imgs, (yn, yt, ys) in train_ld:
            imgs = imgs.to(device, non_blocking=True)
            yn = yn.to(device, non_blocking=True)
            yt = yt.to(device, non_blocking=True)
            ys = ys.to(device, non_blocking=True)

            optimizer.zero_grad()
            loss = loss_fn(model(imgs), (yn, yt, ys))
            loss.backward()
            optimizer.step()
            tr_loss += loss.item() * imgs.size(0)

        tr_loss /= len(train_ds)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, (yn, yt, ys) in val_ld:
                imgs = imgs.to(device, non_blocking=True)
                yn = yn.to(device, non_blocking=True)
                yt = yt.to(device, non_blocking=True)
                ys = ys.to(device, non_blocking=True)
                val_loss += loss_fn(model(imgs), (yn, yt, ys)).item() * imgs.size(0)

        val_loss /= len(val_ds)
        scheduler.step(val_loss)

        print(f"Epoch [{epoch:02d}/{args.epochs}] | Train Loss: {tr_loss:.4f} | Val Loss: {val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), args.save)
            print(f"🔥 Saved best -> {args.save} (val={best_val:.4f})")

    print("🎉 Training complete. Best val loss:", best_val)

if __name__ == "__main__":
    main()
