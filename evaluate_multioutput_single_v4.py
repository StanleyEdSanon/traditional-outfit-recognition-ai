import os
import argparse
import torch
import torch.nn as nn
import pandas as pd
import re

from torch.utils.data import DataLoader
from torchvision import transforms as T
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_recall_fscore_support,
    accuracy_score,
)

from dataset_loader import MultiOutputOutfitDataset

from models.multioutput_resnet import MultiOutputResNet18
from models.multioutput_efficientnet import MultiOutputEfficientNetB0
from models.multioutput_vgg16 import MultiOutputVGG16
from models.multioutput_deit import MultiOutputDeiT

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 224


def build_transform():
    return T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])


def get_model(name: str) -> nn.Module:
    name = name.strip().lower()
    if name == "resnet":
        return MultiOutputResNet18()
    if name == "efficientnet":
        return MultiOutputEfficientNetB0()
    if name == "vgg16":
        return MultiOutputVGG16()
    if name == "deit":
        return MultiOutputDeiT()
    raise ValueError(f"Unknown model: {name}")


# ---------- checkpoint helpers (same spirit as ensemble v4) ----------
def extract_state_dict(ckpt):
    if isinstance(ckpt, dict):
        for k in ["state_dict", "model", "model_state_dict", "net"]:
            if k in ckpt and isinstance(ckpt[k], dict):
                return ckpt[k]
    return ckpt


def remap_common_prefixes(sd, model_keys):
    """
    If the checkpoint uses `features.*` but model expects `backbone.features.*` (or vice-versa),
    map accordingly.
    """
    mapped = {}

    expects_backbone = any(k.startswith("backbone.") for k in model_keys)
    has_backbone = any(str(k).startswith("backbone.") for k in sd.keys())

    for k, v in sd.items():
        ks = str(k)

        # ckpt: features.*  -> model: backbone.features.*
        if expects_backbone and not has_backbone and (ks.startswith("features.") or ks.startswith("classifier.")):
            mapped["backbone." + ks] = v

        # ckpt: backbone.features.* -> model: features.*
        elif (not expects_backbone) and has_backbone and ks.startswith("backbone."):
            mapped[ks[len("backbone."):]] = v

        else:
            mapped[ks] = v

    return mapped


def remap_head_style(sd, model_keys):
    """
    Handles head naming differences:
      - name_head/type_head/sleeve_head
      - head_name/head_type/head_sleeve
    """
    head_a = ("head_name.", "head_type.", "head_sleeve.")
    head_b = ("name_head.", "type_head.", "sleeve_head.")

    has_a = any(str(k).startswith(head_a) for k in sd.keys())
    has_b = any(str(k).startswith(head_b) for k in sd.keys())
    expects_a = any(k.startswith(head_a) for k in model_keys)
    expects_b = any(k.startswith(head_b) for k in model_keys)

    if has_a and expects_b:
        mapped = {}
        for k, v in sd.items():
            k = str(k)
            if k.startswith("head_name."):
                mapped["name_head." + k[len("head_name."):]] = v
            elif k.startswith("head_type."):
                mapped["type_head." + k[len("head_type."):]] = v
            elif k.startswith("head_sleeve."):
                mapped["sleeve_head." + k[len("head_sleeve."):]] = v
            else:
                mapped[k] = v
        return mapped

    if has_b and expects_a:
        mapped = {}
        for k, v in sd.items():
            k = str(k)
            if k.startswith("name_head."):
                mapped["head_name." + k[len("name_head."):]] = v
            elif k.startswith("type_head."):
                mapped["head_type." + k[len("type_head."):]] = v
            elif k.startswith("sleeve_head."):
                mapped["head_sleeve." + k[len("sleeve_head."):]] = v
            else:
                mapped[k] = v
        return mapped

    return sd


def remap_keys(sd, model):
    model_keys = set(model.state_dict().keys())
    sd = remap_common_prefixes(sd, model_keys)
    sd = remap_head_style(sd, model_keys)
    return sd


def strict_clean_load(model, weights_path):
    ckpt = torch.load(weights_path, map_location="cpu")
    sd = extract_state_dict(ckpt)
    sd = remap_keys(sd, model)

    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        print(f"\n[WARN] Load warnings for: {weights_path}")
        print(f"  Missing keys: {len(missing)} (first 10)")
        for k in missing[:10]:
            print("   -", k)
        print(f"  Unexpected keys: {len(unexpected)} (first 10)")
        for k in unexpected[:10]:
            print("   -", k)
        # If backbone is not loading, evaluation is garbage -> hard fail.
        # Heuristic: if a ton of missing keys and they look like features/backbone features, stop.
        bad = [k for k in missing[:50] if ("features." in k or "backbone.features." in k)]
        if len(bad) >= 5:
            raise SystemExit("❌ Backbone not loaded properly. Stop. Fix key mapping / model mismatch.")

    print("✅ Loaded weights:", weights_path)


def audit_csv(csv_path, require_roi=False):
    df = pd.read_csv(csv_path)
    required = {"image", "name_idx", "type_idx", "sleeve_idx"}
    if not required.issubset(df.columns):
        raise SystemExit(f"[FATAL] CSV missing columns. Found: {list(df.columns)}")

    if require_roi:
        roi_pattern = re.compile(r"(^|/|\\)roi_crops($|/|\\)", re.IGNORECASE)
        bad = df[~df["image"].apply(lambda p: bool(roi_pattern.search(str(p))))]
        if len(bad) > 0:
            raise SystemExit(f"[FATAL] Non-ROI path detected: {bad['image'].iloc[0]}")

    missing = df[~df["image"].apply(lambda p: os.path.exists(str(p)))]
    if len(missing) > 0:
        raise SystemExit(f"[FATAL] Missing file: {missing['image'].iloc[0]}")

    print("✅ CSV audit passed")


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    y_true = [[], [], []]
    y_pred = [[], [], []]

    for imgs, (yn, yt, ys) in loader:
        imgs = imgs.to(DEVICE)
        outN, outT, outS = model(imgs)

        pN = outN.argmax(1).cpu().numpy()
        pT = outT.argmax(1).cpu().numpy()
        pS = outS.argmax(1).cpu().numpy()

        y_true[0].extend(yn.numpy()); y_pred[0].extend(pN)
        y_true[1].extend(yt.numpy()); y_pred[1].extend(pT)
        y_true[2].extend(ys.numpy()); y_pred[2].extend(pS)

    return y_true, y_pred


def summarize(head_name, y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    print("\n============================")
    print(head_name)
    print("============================")
    print(f"Accuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, digits=4, zero_division=0))
    print("Confusion Matrix:")
    print(cm)

    return acc, prec, rec, f1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["resnet", "efficientnet", "vgg16", "deit"])
    ap.add_argument("--weights", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--require_roi", action="store_true", help="Fail if csv is not roi_crops/*")
    args = ap.parse_args()

    print("Using device:", DEVICE)
    audit_csv(args.csv, require_roi=args.require_roi)

    tfm = build_transform()
    ds = MultiOutputOutfitDataset(args.csv, transform=tfm)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=False, num_workers=4, pin_memory=True)

    model = get_model(args.model).to(DEVICE)
    strict_clean_load(model, args.weights)

    y_true, y_pred = evaluate(model, dl)

    mN = summarize("OUTFIT NAME", y_true[0], y_pred[0])
    mT = summarize("CLOTHING TYPE", y_true[1], y_pred[1])
    mS = summarize("SLEEVE TYPE", y_true[2], y_pred[2])

    overall_acc = (mN[0] + mT[0] + mS[0]) / 3.0
    thesis_prec = (mN[1] + mT[1] + mS[1]) / 3.0
    thesis_rec  = (mN[2] + mT[2] + mS[2]) / 3.0
    thesis_f1   = (mN[3] + mT[3] + mS[3]) / 3.0

    print("\n============================")
    print("OVERALL ATTRIBUTE ACCURACY")
    print("============================")
    print(f"{overall_acc:.4f}")

    print("\n=== THESIS SUMMARY (macro-avg over 3 heads) ===")
    print(f"Acc : {overall_acc:.4f}")
    print(f"Prec: {thesis_prec:.4f}")
    print(f"Rec : {thesis_rec:.4f}")
    print(f"F1  : {thesis_f1:.4f}")


if __name__ == "__main__":
    main()