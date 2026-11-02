"""Stage 1: explore the raw dataset before building anything.

Three questions: how many images per class, what sizes/colour modes they
come in, and what they actually look like.
Run from the project root:  python src/explore.py
"""

import random
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image

raw_dir = Path("data/raw")
gabungan_dir = Path("data/raw/Dataset_Gabungan")   # both source datasets in one folder - the one used from here on
IMAGE_TYPES = [".jpg", ".png"]


# 1. Image counts per class, for every dataset folder. 4 classes!
# The classes turned out uneven, which is why train.py uses class weights not class names.
for item in sorted(raw_dir.iterdir()):
    if item.is_dir():
        print(f"Dataset folder: {item.name}")
        for class_folder in sorted(item.iterdir()):
            if class_folder.is_dir():
                image_count = sum(1 for f in class_folder.iterdir() if f.suffix.lower() in IMAGE_TYPES)
                print(f"  Class '{class_folder.name}': {image_count} images")


# 2. Image sizes and colour modes per class.
# Mixed sizes are why data.py resizes everything to 224x224, and any
# non-RGB modes (greyscale, RGBA) are why images get .convert("RGB") on load.
for class_folder in sorted(gabungan_dir.iterdir()):
    if not class_folder.is_dir():
        continue
    image_sizes = Counter()
    for image_file in sorted(class_folder.iterdir()):
        if image_file.suffix.lower() in IMAGE_TYPES:
            with Image.open(image_file) as img:
                image_sizes[(img.size, img.mode)] += 1
    print(f"Class folder {class_folder.name}: {image_sizes}")


# 3. A 4 x 6 grid of random examples: one row per class, 6 images per row.
# Looking at the photos before training is we you catch things like
# watermarks, text overlays or one class always having the same background. Open and examine image files
random.seed(42)   # same "random" picks every run

figures_dir = Path("figures")
figures_dir.mkdir(exist_ok=True)

fig, axes = plt.subplots(4, 6, figsize=(18, 12))   #once,before the loop

row = 0
for class_folder in sorted(gabungan_dir.iterdir()):
    if not class_folder.is_dir():
        continue

    # collect this class's image files
    image_files = []
    for f in sorted(class_folder.iterdir()):
        if f.suffix.lower() in IMAGE_TYPES:
            image_files.append(f)

    # pick 6 at random and draw them in this class's row, ouputs 6 in a row
    picks = random.sample(image_files, 6)
    for i, path in enumerate(picks):
        with Image.open(path) as img:
            axes[row, i].imshow(img)
        axes[row, i].set_title(f"{class_folder.name}\n{path.name}", fontsize=8)
        axes[row, i].axis("off")

    row += 1   # next class goes on the next row

plt.tight_layout()
fig.savefig(figures_dir / "class_samples.png", dpi=150)
plt.close(fig)
print("Saved figures/class_samples.png")
