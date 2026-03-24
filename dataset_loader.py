import os
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageOps
import pandas as pd


class MultiOutputOutfitDataset(Dataset):
    def __init__(self, csv_file, transform=None, require_roi=False, root_dir=None):
        """
        csv_file: CSV with columns image,name_idx,type_idx,sleeve_idx
        require_roi: if True, ensures roi_crops paths
        root_dir: optional prefix for relative paths
        """
        self.df = pd.read_csv(csv_file)
        self.transform = transform
        self.require_roi = require_roi
        self.root_dir = root_dir

        self.df["image"] = self.df["image"].astype(str).str.strip()

        if self.root_dir:
            self.df["image"] = self.df["image"].apply(
                lambda p: os.path.join(self.root_dir, p) if not os.path.isabs(p) else p
            )

        if self.require_roi:
            bad = self.df[~self.df["image"].str.contains("roi_crops")]
            if len(bad) > 0:
                raise ValueError(
                    f"[FATAL] Non-ROI paths found. Example: {bad['image'].iloc[0]}"
                )

        missing = self.df[~self.df["image"].apply(os.path.exists)]
        if len(missing) > 0:
            raise FileNotFoundError(
                f"[FATAL] Missing image: {missing['image'].iloc[0]}"
            )

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = row["image"]

        img = Image.open(path).convert("RGB")
        img = ImageOps.exif_transpose(img)

        if self.transform:
            img = self.transform(img)

        name_idx = torch.tensor(int(row["name_idx"]), dtype=torch.long)
        type_idx = torch.tensor(int(row["type_idx"]), dtype=torch.long)
        sleeve_idx = torch.tensor(int(row["sleeve_idx"]), dtype=torch.long)

        return img, (name_idx, type_idx, sleeve_idx)

    def __len__(self):
        return len(self.df)