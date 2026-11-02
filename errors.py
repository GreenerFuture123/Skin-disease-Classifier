"""Stage 6: this stage we find the validation images the fine-tuned models get wrong.

Uses the validation set, not test, so the test set stays untouched until the end.
An image wrong on all 3 seeds is a consistent problem (hard case, or maybe a bad
label) rather than bad luck from one training run.
"""

from collections import Counter

import torch
from torch.utils.data import DataLoader

from data import CLASSES, SkinDataset, eval_transform
from model import build_model

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
val_ds = SkinDataset("val", transform=eval_transform)
val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)   # no shuffle, so prediction i matches row i of val_ds.df

wrong_counts = Counter()   # (true label, group, filename) -> how many seeds got it wrong - think about why when completing this.
for seed in [42, 1, 2]:
    model = build_model()
    model.load_state_dict(torch.load(f"best_finetuned_seed{seed}.pt", map_location=device))
    model.to(device)
    model.eval()

    preds = []
    with torch.no_grad():
        for images, _ in val_loader:
            preds.extend(model(images.to(device)).argmax(dim=1).cpu().tolist())

    for i, pred in enumerate(preds):
        row = val_ds.df.iloc[i]
        if CLASSES[pred] != row["label"]:
            wrong_counts[(row["label"], row["group"], row["path"].split("/")[-1])] += 1

for (label, group, name), times in sorted(wrong_counts.items()):
    print(f"{label:<11} group {group:<4} wrong {times}/3  {name}")
print(f"Images wrong at least once: {len(wrong_counts)}")
print(f"Images wrong on all 3 seeds: {sum(t == 3 for t in wrong_counts.values())}")
#make a note of which files got it wrong 3/3 times. Question why, look at the data after running script in terminal.
