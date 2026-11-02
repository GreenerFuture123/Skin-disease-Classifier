"""Stage 1, Step 4 + Stage 2, Step 1: find near-duplicate images and group them.

Why this matters: public datasets like this often contain the same photo more
than once (slightly different so could be resized, re-saved etc. Or copied between the two source datasets).
If one copy lands in training and another in the test set, the model has
already seen the test image and the score comes out higher than it should.
So duplicates get grouped here, and make_split.py keeps each group together. Doing this means final results are more accurate. 

Run from the project root:  python src/duplicates.py
"""

from itertools import combinations
from pathlib import Path

import imagehash
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

gabungan_dir = Path("data/raw/Dataset_Gabungan")
MAX_DISTANCE = 10   # pHash pairs this close (or closer) count as the same photo - chosen after checking pairs by eye (Piece C)


# A: compute a perceptual hash for every image (1,429 total - check this amount on your own computer, it will change later!).
# A pHash is a 64-bit fingerprint of what the image looks like. Unlike a normal
# file hash it barely changes when a photo is resized or re-compressed, so
# near-copies end up with near-identical hashes which skewes results.

records = []   # only one entry per image: (path, class name, hash)
for class_folder in sorted(gabungan_dir.iterdir()):
    if not class_folder.is_dir():
        continue
    for f in sorted(class_folder.iterdir()):
        if f.suffix.lower() in [".jpg", ".png"]:
            with Image.open(f) as img:
                h = imagehash.phash(img)
            records.append((f, class_folder.name, h))

print(f"Hashed {len(records)} images")
print(records[0])


# B: compare every pair of images.
# hash_a - hash_b gives the Hamming distance: how many of the 64 bits differ.
# 0 = identical fingerprint, small numbers = very likely the same photo.
# (~1 million pairs, so this may take a little while.)

close_pairs = []
for rec_a, rec_b in combinations(records, 2):
    path_a, class_a, hash_a = rec_a
    path_b, class_b, hash_b = rec_b
    dist = hash_a - hash_b
    if dist <= MAX_DISTANCE:
        close_pairs.append((rec_a, rec_b, dist))

print(f"Pairs with distance <= {MAX_DISTANCE}: {len(close_pairs)}")

# How many pairs at each threshold, and how many of those have DIFFERENT labels.
# Different labels on the same photo = a labelling conflict that needs a decision.
for threshold in [0, 2, 4, 6, 8, 10]:
    n_pairs = sum(1 for a, b, d in close_pairs if d <= threshold)
    n_cross = sum(1 for a, b, d in close_pairs if d <= threshold and a[1] != b[1])
    print(f"  <= {threshold:2}: {n_pairs:4} pairs, {n_cross:3} with different labels")


# C: draw the borderline pairs side by side to check by eye. Open each file that appears after running and note down file No and Name. 
# Borderline = different labels, or distance 6-10 (close, but not certain).
# This is what the MAX_DISTANCE choice and the conflict list in make_split.py are based on...

figures_dir = Path("figures")
figures_dir.mkdir(exist_ok=True)

to_check = []
for a, b, d in close_pairs:
    if a[1] != b[1] or d >= 6:
        to_check.append((a, b, d))
print(f"Pairs to check by eye: {len(to_check)}")

if to_check:
    # one row per pair, 2 columns (A and B)
    # squeeze=False keeps axes 2D even if there's only one pair to show
    fig, axes = plt.subplots(len(to_check), 2, figsize=(6, 3 * len(to_check)), squeeze=False)
    for row, (a, b, d) in enumerate(to_check):
        for col, rec in enumerate([a, b]):
            with Image.open(rec[0]) as img:
                axes[row, col].imshow(img)
            axes[row, col].set_title(f"{rec[0].name}\n{rec[1]}  (d={d})", fontsize=8)
            axes[row, col].axis("off")
    plt.tight_layout()
    fig.savefig(figures_dir / "duplicate_pairs.png")
    plt.close(fig)


# Stage 2, Step 1: merge close pairs into duplicate groups.
# Groups chain together: if A matches B and B matches C, all three end up in
# one group, even if A and C aren't close to each other themselves.

# 1. every image starts in its own group
group_of = {}
for i, (path, label, h) in enumerate(records):
    group_of[path] = i

# 2. for each close pair, move everything in B's group into A's group - separates data cleanly. 
for a, b, d in close_pairs:
    group_a = group_of[a[0]]
    group_b = group_of[b[0]]
    if group_a != group_b:
        for p in group_of:
            if group_of[p] == group_b:
                group_of[p] = group_a

print(f"Groups: {len(set(group_of.values()))}")

# 3. one row per image, saved for make_split.py
rows = []
for path, label, h in records:
    rows.append({
        "path": str(path),
        "label": label,
        "source": path.name[:3],   # "ds1" or "ds2" - which original dataset it came from(check names if renamed)
        "group": group_of[path],
    })
pd.DataFrame(rows).to_csv("data/groups.csv", index=False)
print("Saved data/groups.csv")
