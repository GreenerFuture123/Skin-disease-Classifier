"""Stage 3: input pipeline - turns rows of data/splits.csv into tensors for the model to use.

Quick check from the project root:  python src/data.py
"""

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2


# The order here sets the label numbers (Chickenpox = 0, Measles = 1, ...).
# Every other script imports this list, so the numbers match everywhere. Check same order throughout! 
CLASSES = ["Chickenpox", "Measles", "Monkeypox", "Normal"]
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASSES)}


# --- Transforms ---
# MobileNetV2 was pretrained on ImageNet, so inputs should be 224x224 and
# normalised with ImageNet's mean/std - the pretrained filters expect that scale.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Val/test: no randomness here so the same image always gets the same score.
eval_transform = v2.Compose([
    v2.Resize((224, 224)),                   # squash to square rather than crop, so lesions near the edge aren't cut off, more accurate
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),   # 0-255 ints -> 0.0-1.0 floats
    v2.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

# Training: small random changes so the model sees a slightly different version
# of each photo every epoch - helps with overfitting on ~1,000 training images.
# Kept mild on purpose :
#   - flips and small rotations don't change what a rash looks like (core image is still the same) 
#   - brightness only, no hue/saturation jitter, because colour (redness etc.) Tried using hue and saturation - didnt work.see below 
#     the colour amongst other things is what tells these conditions apart
train_transform = v2.Compose([
    v2.Resize((224, 224)),
    v2.RandomHorizontalFlip(p=0.5),
    v2.RandomRotation(degrees=15),
    v2.ColorJitter(brightness=0.2),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


def apply_clahe(image):
    """Optional contrast boost (CLAHE), switched on with USE_CLAHE in train.py.

    Works on the lightness channel only, so it sharpens
    local contrast without shifting the skin colours or background colour. 
    """
    lab = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    return Image.fromarray(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB))


class SkinDataset(Dataset):
    """One split (train / val / test) of data/splits.csv.

    Images are loaded from disk one at a time when asked for, rather than
    all at once, so memory usage stays low throughout development
    """

    def __init__(self, split, csv_path="data/splits.csv", transform=None, use_clahe=False):
        df = pd.read_csv(csv_path)
        self.df = df[df["split"] == split].reset_index(drop=True)   # reset so row 0..N-1 lines up with item 
        self.transform = transform
        self.use_clahe = use_clahe

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        image = Image.open(row["path"]).convert("RGB")   # some files aren't RGB, some are (see other file - explore.py)
        if self.use_clahe:
            image = apply_clahe(image)
        label = CLASS_TO_IDX[row["label"]]                # class name <<->> number
        if self.transform:
            image = self.transform(image)
        return image, label


if __name__ == "__main__":
    # Quick check before moving forward (seebelow also): one image should be [3, 224, 224] float32, roughly centred on 0
    train_ds = SkinDataset("train", transform=train_transform)
    print(f"Train images: {len(train_ds)}")
    image, label = train_ds[0]
    print(image.shape, image.dtype, image.min(), image.max(), label, CLASSES[label])

    # and one batch should be [32, 3, 224, 224] with 32 labels
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    images, labels = next(iter(train_loader))
    print(images.shape, labels.shape)
    print(labels)
