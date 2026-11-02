"""Stage 5: train the model in two phases - frozen base first, then fine-tuning.

Run once per seed (42, 1, 2) by changing SEED below. Three seeds show how much
the result depends on luck (shuffle order, augmentation) rather than the method.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

SEED = 42
USE_CLAHE = False                               # True = run the CLAHE experiment (see data.py file)
RUN_NAME = "clahe_" if USE_CLAHE else ""        # keeps CLAHE checkpoints separate from the normal ones
torch.manual_seed(SEED)                         # set before anything  random happens

from data import CLASSES, SkinDataset, train_transform, eval_transform
from model import build_model

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")   # Apple GPU if available

train_ds = SkinDataset("train", transform=train_transform, use_clahe=USE_CLAHE)
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_ds = SkinDataset("val", transform=eval_transform, use_clahe=USE_CLAHE)
val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

model = build_model().to(device)

# Class weights: the classes aren't the same size, so without weights the model
# can score well by mostly guessing the biggest class. Each class's weight is
# (total images) / (num classes * images in that class), so rarer classes count
# as more in the losses. Worked out from the training split only.
label_counts = train_ds.df["label"].value_counts()
counts = torch.tensor([label_counts[name] for name in CLASSES], dtype=torch.float32)
weights = counts.sum() / (len(CLASSES) * counts)
print(dict(zip(CLASSES, weights.tolist())))

criterion = nn.CrossEntropyLoss(weight=weights.to(device))   # training loss (!weighted)
val_criterion = nn.CrossEntropyLoss()                         # validation loss (!unweighted)


def run_phase(lr, num_epochs, save_path):
    """Train for num_epochs, keep whichever epoch had the lowest val loss."""
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=lr   # only for the unfrozen layers
    )
    best_val_loss = float("inf")
    for epoch in range(num_epochs):
        # --- Train ---
        model.train()            # dropout on
        model.features.eval()    # but BatchNorm in the pretrained base stays in evaluation mode:
                                 # with small batches its running stats would get noisy and
                                 # undo some of the ImageNet pretraining
        total_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()              # clear last batch's gradients
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()                    # work out how each parameter should change then 
            optimizer.step()                   # apply those changes

            total_loss += loss.item()

        # --- Validate model ---
        model.eval()
        val_loss = 0
        correct = 0
        with torch.no_grad():                  # no gradients needed, = faster + less memory
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                val_loss += val_criterion(outputs, labels).item()
                preds = outputs.argmax(dim=1)  # highest-scoring class for each image
                correct += (preds == labels).sum().item()

        # --- Report best epoch then save if this is the best epoch so far ---
        avg_train_loss = total_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch + 1}: train {avg_train_loss:.4f} | val {avg_val_loss:.4f} | acc {correct / len(val_ds):.3f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), save_path)

    # end the phase on the best epoch, not the last one
    model.load_state_dict(torch.load(save_path, map_location=device))


# 1: only the new final layer learns. Higher learning rate is fine
# because it's a single fresh layer starting from random weights.
print("Phase 1: frozen base")
run_phase(lr=1e-3, num_epochs=10, save_path=f"best_frozen_{RUN_NAME}seed{SEED}.pt")

# 2: also unfreeze the last 4 blocks of the base so they can adapt to
# skin images. Learning rate 10x lower so the pretrained weights get nudged but
# not overwritten. The early blocks (edges, basic textures) stay frozen.
print("Phase 2: fine-tuning")
for param in model.features[-4:].parameters():
    param.requires_grad = True
print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
run_phase(lr=1e-4, num_epochs=10, save_path=f"best_finetuned_{RUN_NAME}seed{SEED}.pt")
