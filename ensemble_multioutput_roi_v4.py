import os
import re
import argparse
import torch
import torch.nn as nn
import pandas as pd

from torch.utils.data import DataLoader
from torchvision import transforms as T
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score, classification_report

from dataset_loader import MultiOutputOutfitDataset

from models.multioutput_resnet import MultiOutputResNet18
from models.multioutput_efficientnet import MultiOutputEfficientNetB0
from models.multioutput_vgg16 import MultiOutputVGG16
from models.multioutput_deit import MultiOutputDeiT


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 224
IMG_EXT = (".jpg", ".jpeg", ".png")


# ---------------- TRANSFORM ----------------
def build_transform():
    return T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])


# ---------------- MODEL FACTORY ----------------
def get_model(tag: str) -> nn.Module:
    tag = tag.strip().lower()
    if tag == "resnet":
        return MultiOutputResNet18()
    if tag == "efficientnet":
        return MultiOutputEfficientNetB0()
    if tag == "vgg16":
        return MultiOutputVGG16()
    if tag == "deit":
        return MultiOutputDeiT()
    raise ValueError(f"[FATAL] Unknown model tag: {tag}. Use one of: resnet, efficientnet, vgg16, deit")


# ---------------- CHECKPOINT HELPERS ----------------
def extract_state_dict(ckpt_obj):
    """Handle raw state_dict or wrapped checkpoints."""
    if isinstance(ckpt_obj, dict):
        for k in ["state_dict", "model", "model_state_dict", "net"]:
            if k in ckpt_obj and isinstance(ckpt_obj[k], dict):
                return ckpt_obj[k]
    return ckpt_obj


def _needs_prefix(model_keys, prefix: str) -> bool:
    return any(k.startswith(prefix) for k in model_keys)


def remap_backbone_prefixes(sd: dict, model_keys: set) -> dict:
    """
    Fix common backbone prefix mismatches:
      - checkpoint uses "features.*" but model expects "backbone.features.*"
      - checkpoint uses "classifier.*" but model expects "backbone.classifier.*"
    """
    expects_backbone_features = _needs_prefix(model_keys, "backbone.features.")
    expects_backbone_classifier = _needs_prefix(model_keys, "backbone.classifier.")
    expects_plain_features = _needs_prefix(model_keys, "features.")
    expects_plain_classifier = _needs_prefix(model_keys, "classifier.")

    out = {}
    for k, v in sd.items():
        ks = str(k)

        # ckpt: features.*  -> model: backbone.features.*
        if expects_backbone_features and ks.startswith("features."):
            out["backbone." + ks] = v
            continue

        # ckpt: backbone.features.* -> model: features.*
        if expects_plain_features and ks.startswith("backbone.features."):
            out[ks[len("backbone."):]] = v
            continue

        # ckpt: classifier.* -> model: backbone.classifier.*
        if expects_backbone_classifier and ks.startswith("classifier."):
            out["backbone." + ks] = v
            continue

        # ckpt: backbone.classifier.* -> model: classifier.*
        if expects_plain_classifier and ks.startswith("backbone.classifier."):
            out[ks[len("backbone."):]] = v
            continue

        out[ks] = v

    return out


def remap_head_style(sd: dict, model_keys: set) -> dict:
    """
    Support head naming differences:
      Style A: head_name/head_type/head_sleeve
      Style B: name_head/type_head/sleeve_head
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
            ks = str(k)
            if ks.startswith("head_name."):
                mapped["name_head." + ks[len("head_name."):]] = v
            elif ks.startswith("head_type."):
                mapped["type_head." + ks[len("head_type."):]] = v
            elif ks.startswith("head_sleeve."):
                mapped["sleeve_head." + ks[len("head_sleeve."):]] = v
            else:
                mapped[ks] = v
        return mapped

    if has_b and expects_a:
        mapped = {}
        for k, v in sd.items():
            ks = str(k)
            if ks.startswith("name_head."):
                mapped["head_name." + ks[len("name_head."):]] = v
            elif ks.startswith("type_head."):
                mapped["head_type." + ks[len("type_head."):]] = v
            elif ks.startswith("sleeve_head."):
                mapped["head_sleeve." + ks[len("sleeve_head."):]] = v
            else:
                mapped[ks] = v
        return mapped

    return sd


def remap_keys(sd: dict, model: nn.Module) -> dict:
    model_keys = set(model.state_dict().keys())
    sd = remap_backbone_prefixes(sd, model_keys)
    sd = remap_head_style(sd, model_keys)
    return sd


def strict_clean_load(model: nn.Module, weights_path: str):
    """
    Loads weights with automatic remapping, but refuses to proceed if load is not clean.
    (This prevents the 'everything predicts one class' garbage.)
    """
    ckpt = torch.load(weights_path, map_location="cpu")
    sd = extract_state_dict(ckpt)
    if not isinstance(sd, dict):
        raise SystemExit(f"[FATAL] weights file is not a state_dict: {weights_path}")

    sd = remap_keys(sd, model)

    # shape check before load_state_dict
    msd = model.state_dict()
    for k, v in sd.items():
        if k in msd and hasattr(v, "shape") and msd[k].shape != v.shape:
            raise SystemExit(
                f"[FATAL] Shape mismatch for key '{k}': ckpt {tuple(v.shape)} vs model {tuple(msd[k].shape)}"
            )

    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        print(f"\n[WARN] Load warnings for: {weights_path}")
        print(f"  Missing keys: {len(missing)} (first 15)")
        for k in missing[:15]:
            print("   -", k)
        print(f"  Unexpected keys: {len(unexpected)} (first 15)")
        for k in unexpected[:15]:
            print("   -", k)
        raise SystemExit("❌ BAD LOAD — architecture/keys mismatch. Fix it; don't fake results.")

    print(f"✅ Clean load: {weights_path}")


# ---------------- DATASET AUDIT ----------------
def audit_csv(csv_path: str, require_roi: bool):
    df = pd.read_csv(csv_path)
    required = {"image", "name_idx", "type_idx", "sleeve_idx"}
    if not required.issubset(df.columns):
        raise SystemExit(f"[FATAL] CSV missing columns. Found: {list(df.columns)}")

    print("\nCSV sample paths:")
    print(df["image"].head(5).to_string(index=False))

    if require_roi:
        roi_pattern = re.compile(r"(^|/|\\)roi_crops($|/|\\)", re.IGNORECASE)
        bad = df[~df["image"].astype(str).apply(lambda p: bool(roi_pattern.search(str(p))))]
        if len(bad) > 0:
            raise SystemExit(f"[FATAL] Non-ROI path detected: {bad['image'].iloc[0]}")

    missing = df[~df["image"].astype(str).apply(os.path.exists)]
    if len(missing) > 0:
        raise SystemExit(f"[FATAL] Missing file: {missing['image'].iloc[0]}")

    print("✅ CSV audit passed")


# ---------------- WEIGHTS PARSING ----------------
def parse_head_weights(s: str, n_models: int):
    """
    Optional format:
      --head_weights "name=0.7,0.3;type=0.8,0.2;sleeve=0.4,0.6"
    If omitted: equal weights for all models for all heads.
    """
    default = {
        "name":   [1.0 / n_models] * n_models,
        "type":   [1.0 / n_models] * n_models,
        "sleeve": [1.0 / n_models] * n_models,
    }
    if not s:
        return default

    out = {}
    parts = [p.strip() for p in s.split(";") if p.strip()]
    for p in parts:
        if "=" not in p:
            raise SystemExit(f"[FATAL] bad --head_weights chunk: '{p}' (missing '=')")
        k, v = p.split("=", 1)
        k = k.strip().lower()
        ws = [float(x) for x in v.split(",") if x.strip() != ""]
        if len(ws) != n_models:
            raise SystemExit(f"[FATAL] '{k}' has {len(ws)} weights but you provided {n_models} models.")
        sm = sum(ws)
        if sm <= 0:
            raise SystemExit(f"[FATAL] '{k}' weights sum to 0.")
        out[k] = [w / sm for w in ws]

    for k in ["name", "type", "sleeve"]:
        if k not in out:
            out[k] = default[k]
    return out


# ---------------- TEMPERATURE SCALING ----------------
def _fit_temperature(logits_cpu: torch.Tensor, y_cpu: torch.Tensor, max_iter=50) -> float:
    """
    Fit temperature T for a single head using NLL on validation logits.
    """
    logT = torch.zeros(1, requires_grad=True)  # log(T)
    nll = nn.CrossEntropyLoss()
    optimizer = torch.optim.LBFGS([logT], lr=0.5, max_iter=max_iter)

    def closure():
        optimizer.zero_grad()
        Tval = torch.exp(logT)
        loss = nll(logits_cpu / Tval, y_cpu)
        loss.backward()
        return loss

    optimizer.step(closure)
    Tval = torch.exp(logT).item()
    return float(max(Tval, 1e-6))


@torch.no_grad()
def collect_val_logits(model, loader):
    model.eval()
    ln, lt, ls = [], [], []
    yn, yt, ys = [], [], []

    for imgs, (y1, y2, y3) in loader:
        imgs = imgs.to(DEVICE)
        o1, o2, o3 = model(imgs)

        ln.append(o1.detach().cpu())
        lt.append(o2.detach().cpu())
        ls.append(o3.detach().cpu())

        yn.append(y1.detach().cpu())
        yt.append(y2.detach().cpu())
        ys.append(y3.detach().cpu())

    return torch.cat(ln, 0), torch.cat(lt, 0), torch.cat(ls, 0), torch.cat(yn, 0), torch.cat(yt, 0), torch.cat(ys, 0)


def calibrate_models(models, val_loader):
    temps = []
    print("\n=== Temperature Scaling on VAL ===")
    for i, m in enumerate(models):
        ln, lt, ls, yn, yt, ys = collect_val_logits(m, val_loader)
        Tn = _fit_temperature(ln, yn)
        Tt = _fit_temperature(lt, yt)
        Ts = _fit_temperature(ls, ys)
        temps.append({"name": Tn, "type": Tt, "sleeve": Ts})
        print(f"Model {i}: T_name={Tn:.4f}, T_type={Tt:.4f}, T_sleeve={Ts:.4f}")
    return temps


# ---------------- ENSEMBLE / SINGLE EVAL ----------------
@torch.no_grad()
def eval_models(models, loader, head_w, temps=None):
    """
    If len(models)=1 this is just normal evaluation.
    Otherwise, soft-voting ensemble by weighted probability sum.
    """
    all_true = [[], [], []]
    all_pred = [[], [], []]

    for imgs, (y_name, y_type, y_sleeve) in loader:
        imgs = imgs.to(DEVICE)

        probs_name = None
        probs_type = None
        probs_sleeve = None

        for mi, m in enumerate(models):
            o1, o2, o3 = m(imgs)

            if temps is not None:
                o1 = o1 / temps[mi]["name"]
                o2 = o2 / temps[mi]["type"]
                o3 = o3 / temps[mi]["sleeve"]

            pn = torch.softmax(o1, dim=1).cpu() * head_w["name"][mi]
            pt = torch.softmax(o2, dim=1).cpu() * head_w["type"][mi]
            ps = torch.softmax(o3, dim=1).cpu() * head_w["sleeve"][mi]

            probs_name = pn if probs_name is None else probs_name + pn
            probs_type = pt if probs_type is None else probs_type + pt
            probs_sleeve = ps if probs_sleeve is None else probs_sleeve + ps

        pred_name = probs_name.argmax(dim=1).numpy()
        pred_type = probs_type.argmax(dim=1).numpy()
        pred_sleeve = probs_sleeve.argmax(dim=1).numpy()

        truths = [y_name.numpy(), y_type.numpy(), y_sleeve.numpy()]
        preds = [pred_name, pred_type, pred_sleeve]

        for i in range(3):
            all_true[i].extend(truths[i])
            all_pred[i].extend(preds[i])

    return all_true, all_pred


def summarize(y_true, y_pred, title, show_report=False):
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred)

    print("\n============================")
    print(title)
    print("============================")
    print(f"Accuracy       : {acc:.4f}")
    print(f"Macro Precision: {prec:.4f}")
    print(f"Macro Recall   : {rec:.4f}")
    print(f"Macro F1       : {f1:.4f}")

    if show_report:
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, digits=4, zero_division=0))

    print("\nConfusion Matrix:\n", cm)

    return acc, prec, rec, f1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", required=True, help="Comma-separated model tags: resnet,efficientnet,vgg16,deit")
    parser.add_argument("--weights", required=True, help="Comma-separated .pt paths, same order as --models")
    parser.add_argument("--csv", required=True, help="Test CSV (ROI or full depending on --require_roi)")
    parser.add_argument("--require_roi", action="store_true", help="Enforce roi_crops paths in CSV")
    parser.add_argument("--val_csv", default=None, help="Validation CSV (for --calibrate)")
    parser.add_argument("--calibrate", action="store_true", help="Enable temperature scaling using --val_csv")
    parser.add_argument("--head_weights", default="", help='e.g. "name=0.7,0.3;type=0.8,0.2;sleeve=0.4,0.6"')
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--show_report", action="store_true", help="Print sklearn classification_report per head")
    parser.add_argument("--latex_name", default="Ensemble")
    args = parser.parse_args()

    print("Using device:", DEVICE)

    audit_csv(args.csv, require_roi=args.require_roi)
    if args.calibrate:
        if not args.val_csv:
            raise SystemExit("[FATAL] --calibrate requires --val_csv")
        audit_csv(args.val_csv, require_roi=args.require_roi)

    transform = build_transform()

    test_ds = MultiOutputOutfitDataset(args.csv, transform=transform, require_roi=args.require_roi)
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    model_tags = [m.strip() for m in args.models.split(",") if m.strip()]
    weight_paths = [w.strip() for w in args.weights.split(",") if w.strip()]
    if len(model_tags) != len(weight_paths):
        raise SystemExit("[FATAL] models and weights count mismatch")

    models_list = []
    for tag, wp in zip(model_tags, weight_paths):
        m = get_model(tag)
        strict_clean_load(m, wp)
        m.to(DEVICE)
        m.eval()
        models_list.append(m)

    head_w = parse_head_weights(args.head_weights, n_models=len(models_list))
    print("\n=== Head Weights (normalized) ===")
    print("name  :", head_w["name"])
    print("type  :", head_w["type"])
    print("sleeve:", head_w["sleeve"])

    temps = None
    if args.calibrate:
        val_ds = MultiOutputOutfitDataset(args.val_csv, transform=transform, require_roi=args.require_roi)
        val_loader = DataLoader(
            val_ds,
            batch_size=args.batch,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True
        )
        temps = calibrate_models(models_list, val_loader)

    y_true, y_pred = eval_models(models_list, test_loader, head_w=head_w, temps=temps)

    titles = ["OUTFIT NAME", "CLOTHING TYPE", "SLEEVE TYPE"]
    metrics = [summarize(y_true[i], y_pred[i], titles[i], show_report=args.show_report) for i in range(3)]

    accs  = [m[0] for m in metrics]
    precs = [m[1] for m in metrics]
    recs  = [m[2] for m in metrics]
    f1s   = [m[3] for m in metrics]

    overall_acc = sum(accs) / 3.0
    thesis_prec = sum(precs) / 3.0
    thesis_rec  = sum(recs) / 3.0
    thesis_f1   = sum(f1s) / 3.0

    print("\n============================")
    print("OVERALL ATTRIBUTE ACCURACY")
    print("============================")
    print(f"{overall_acc:.4f}")

    print("\n=== THESIS SUMMARY (macro-avg over 3 heads) ===")
    print(f"Acc : {overall_acc:.4f}")
    print(f"Prec: {thesis_prec:.4f}")
    print(f"Rec : {thesis_rec:.4f}")
    print(f"F1  : {thesis_f1:.4f}")

    print("\nLaTeX row:")
    print(f"{args.latex_name} & {overall_acc:.3f} & {thesis_prec:.3f} & {thesis_rec:.3f} & {thesis_f1:.3f} \\\\")


if __name__ == "__main__":
    main()