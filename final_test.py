"""Final evaluation: the 3 saved models on the test set.

The test set is only used once, here, after every decision (model, augmentation,
CLAHE or not, epochs) was made on the validation set. Using it earlier and
tweaking based on the result would make the final number too optimistic.
"""

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader

from data import CLASSES, CLASS_TO_IDX, SkinDataset, eval_transform
from model import build_model

SPLIT = "test"
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
ds = SkinDataset(SPLIT, transform=eval_transform)
loader = DataLoader(ds, batch_size=32, shuffle=False)   # no shuffle, so predictions line up with ds.df rows
labels = [CLASS_TO_IDX[name] for name in ds.df["label"]]

# "Unique photos": keep one image per duplicate group. Groups can't cross splits,
# but a group can still have several copies inside the test set, which would
# count the same photo more than once. This gives a score where each photo only counts once.
keep = ds.df.drop_duplicates(["group", "label"]).index
print(f"Split: {SPLIT} | images: {len(ds)} | unique photos: {len(keep)}")

accs, f1s, photo_f1s = [], [], []
for seed in [42, 1, 2]:
    model = build_model()
    model.load_state_dict(torch.load(f"best_finetuned_seed{seed}.pt", map_location=device))
    model.to(device)
    model.eval()

    preds = []
    with torch.no_grad():
        for images, _ in loader:
            preds.extend(model(images.to(device)).argmax(dim=1).cpu().tolist())

    print(f"\n=== Seed {seed} ===")
    print(classification_report(labels, preds, target_names=CLASSES, digits=3))
    print(confusion_matrix(labels, preds))
    accs.append(accuracy_score(labels, preds))
    # macro-F1 = average F1 across the 4 classes, each counted equally,
    # so a small class like Measles matters as much as Normal (accuracy alone hides that)
    f1s.append(f1_score(labels, preds, average="macro"))
    photo_f1s.append(f1_score([labels[i] for i in keep], [preds[i] for i in keep], average="macro"))

# mean +/- standard deviation across the 3 seeds, note the 3 seeds used. 
print("\n=== Summary ===")
print(f"Accuracy: {np.mean(accs):.3f} +/- {np.std(accs, ddof=1):.3f}")
print(f"Macro-F1 (images): {np.mean(f1s):.3f} +/- {np.std(f1s, ddof=1):.3f}")
print(f"Macro-F1 (unique photos): {np.mean(photo_f1s):.3f} +/- {np.std(photo_f1s, ddof=1):.3f}")
