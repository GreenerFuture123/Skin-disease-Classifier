"""Plot the test-set confusion matrix.

The numbers are copied from results/test_results.txt (output of final_test.py),
so the test set doesn't have to be run again just to make a figure.
"""

import matplotlib.pyplot as plt
import numpy as np

CLASSES = ["Chickenpox", "Measles", "Monkeypox", "Normal"]

# rows = true class, columns = predicted class (same order as CLASSES mentioned previously - note!)
matrices = np.array([
    [[18, 2, 6, 3], [0, 24, 0, 1], [10, 6, 60, 2], [0, 0, 0, 72]],   # seed 42
    [[26, 0, 0, 3], [0, 24, 0, 1], [10, 3, 63, 2], [0, 1, 0, 71]],   # seed 1
    [[21, 1, 4, 3], [0, 24, 0, 1], [10, 6, 60, 2], [0, 1, 0, 71]],   # seed 2
])

# Add the 3 seeds together, then turn each row into percentages-see below-simple formula.
# Percentages per true class make small and large classes easy to compare.
total = matrices.sum(axis=0)
percent = 100 * total / total.sum(axis=1, keepdims=True)

fig, ax = plt.subplots(figsize=(6, 5))
ax.imshow(percent, cmap="Blues", vmin=0, vmax=100)
ax.set_xticks(range(4), CLASSES)
ax.set_yticks(range(4), CLASSES)
ax.set_xlabel("Predicted class")
ax.set_ylabel("True class")
ax.set_title("Test-set predictions, 3 seeds combined\n(% of each true class)", fontsize=11)

# write the % in each cell (white text on dark cells so its still readable)
for r in range(4):
    for c in range(4):
        value = percent[r, c]
        ax.text(c, r, f"{value:.0f}%", ha="center", va="center",
                color="white" if value > 50 else "#1f2937")

# white grid lines between cells, no tick or border
ax.set_xticks(np.arange(-0.5, 4), minor=True)
ax.set_yticks(np.arange(-0.5, 4), minor=True)
ax.grid(which="minor", color="white", linewidth=2)
ax.tick_params(which="both", length=0)
for spine in ax.spines.values():
    spine.set_visible(False)

fig.tight_layout()
fig.savefig("figures/test_confusion_matrix.png", dpi=150)
print("Saved figures/test_confusion_matrix.png")
