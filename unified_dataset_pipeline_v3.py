import os
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Base path of your dataset
base_path = "/home/e814/Documents/outfit_dataset"

# Collect all records
records = []

for outfit_name in os.listdir(base_path):
    outfit_path = os.path.join(base_path, outfit_name)
    if not os.path.isdir(outfit_path):
        continue

    for outfit_type in os.listdir(outfit_path):
        type_path = os.path.join(outfit_path, outfit_type)
        if not os.path.isdir(type_path):
            continue

        # ✅ Walk through all sleeve_type subfolders
        for sleeve_type in os.listdir(type_path):
            sleeve_path = os.path.join(type_path, sleeve_type)
            if not os.path.isdir(sleeve_path):
                continue

            for img in os.listdir(sleeve_path):
                if img.lower().endswith((".jpg", ".jpeg", ".png")):
                    records.append({
                        "image": os.path.join(sleeve_path, img),
                        "outfit_name": outfit_name,
                        "outfit_type": outfit_type,
                        "sleeve_type": sleeve_type
                    })

# Create DataFrame
df = pd.DataFrame(records)
print(f"🔍 Total images: {len(df)}")

# --- Split into train/val/test ---
train_df, temp_df = train_test_split(df, test_size=0.15, stratify=df["outfit_name"], random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df["outfit_name"], random_state=42)

print(f"Train size: {len(train_df)}")
print(f"Val size:   {len(val_df)}")
print(f"Test size:  {len(test_df)}")

# --- Save multilabel CSVs ---
train_df.to_csv(os.path.join(base_path, "train_multilabel.csv"), index=False)
val_df.to_csv(os.path.join(base_path, "val_multilabel.csv"), index=False)
test_df.to_csv(os.path.join(base_path, "test_multilabel.csv"), index=False)
print("✅ Saved multilabel CSVs")

# --- Convert to numeric CSVs ---
all_outfit_names = sorted(df["outfit_name"].unique())
all_outfit_types = sorted(df["outfit_type"].unique())
all_sleeve_types = sorted(df["sleeve_type"].unique())

def convert_to_numeric(df, path):
    df_num = df.copy()
    for c in all_outfit_names + all_outfit_types + all_sleeve_types:
        df_num[c] = (df_num["outfit_name"].eq(c) |
                     df_num["outfit_type"].eq(c) |
                     df_num["sleeve_type"].eq(c)).astype(int)
    df_num.to_csv(path, index=False)

convert_to_numeric(train_df, os.path.join(base_path, "train_numeric.csv"))
convert_to_numeric(val_df, os.path.join(base_path, "val_numeric.csv"))
convert_to_numeric(test_df, os.path.join(base_path, "test_numeric.csv"))
print("✅ Saved numeric CSVs")

# --- Save distribution plots ---
plot_dir = os.path.join(base_path, "results", "distribution_plots")
os.makedirs(plot_dir, exist_ok=True)

for col in ["outfit_name", "outfit_type", "sleeve_type"]:
    counts = df[col].value_counts()
    counts.plot(kind="bar", title=f"Distribution of {col}", figsize=(8, 6))
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"{col}_distribution.png"))
    plt.close()
    print(f"📊 Saved plot for {col} distribution")
