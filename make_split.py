"""Stage 2, Step 2: split duplicate groups into train / val / test.

Reads data/groups.csv (made by src/duplicates.py) and writes data/splits.csv.
Run from the project root:  python src/make_split.py
"""

# Splitting by GROUP!, not by image: every copy of a photo goes into the same
# split so the test set never contains a near-copy of a training image. If it did, it would produce inaccurate results because t trained with that image.
#
# StratifiedGroupKFold does two jobs at once:
#   1st - group-safe: a whole duplicate group goes to one fold
#   2nd - stratified: each fold keeps roughly the same class mix as the full dataset
# 7 folds -> fold 0 = test (~14%), fold 1 = val (~14%), the other 5 = train (~72%).

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

df = pd.read_csv("data/groups.csv")

# 1. Drop the labelling conflicts.
# These photos appeared in duplicate pairs with different labels (ones checked by eye
# in Stage 1). There's no way to know which label is right, so they're removed
# rather than guessing.
conflict_files = ["ds1_ChickenPox_88.jpg", "ds2_chickenpox94.png",
                  "ds1_MonkeyPox_82.jpg", "ds2_monkeypox84.png"]
filenames = df["path"].str.split("/").str[-1]
df = df[~filenames.isin(conflict_files)].reset_index(drop=True)
print(f"Images after dropping conflicts: {len(df)}")

# 2. Seven group-safe, stratified folds
sgkf = StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=42)   # fixed random_state = same split every run
folds = list(sgkf.split(df, df["label"], df["group"]))                  # (data, labels, groups)

df["split"] = "train"                       # everything starts as train as this is training stage
df.loc[folds[0][1], "split"] = "test"       # ...then fold 0 becomes test
df.loc[folds[1][1], "split"] = "val"        # ...and fold 1 becomes val

# 3. Check the class balance in each split
print(pd.crosstab(df["split"], df["label"], margins=True))

# 4. Sanity checks - stop here if anything is wrong, rather than training on a bad split
assert df.groupby("group")["split"].nunique().max() == 1, "Some groups appear in more than one split"
assert not df["path"].str.split("/").str[-1].isin(conflict_files).any(), "Some conflict files are still in the dataset"
assert df["split"].isin(["train", "val", "test"]).all(), "Some images do not have a valid split"
assert len(df) == 1425, f"Expected 1425 images (1429 - 4 conflicts), got {len(df)}"

df.to_csv("data/splits.csv", index=False)
print("Saved data/splits.csv")
#check spelling for this row if copying, I had multile errors when writing it. 
