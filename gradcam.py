"""Stage 6: Grad-CAM heatmaps on validation images, to check for shortcuts.

A model can score well for the wrong reasons, e.g. recognising the background,
lighting or camera of one source instead of the skin itself. Grad-CAM shows
which parts of the image pushed the model towards its prediction, so you can
check it's actually looking at the lesions.
"""

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from PIL import Image

from data import CLASSES, SkinDataset, eval_transform
from model import build_model

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

model = build_model()
model.load_state_dict(torch.load("best_finetuned_seed42.pt", map_location=device))
model.to(device)
model.eval()

# build_model() freezes the base again, but Grad-CAM needs gradients flowing
# back through it to the last feature map, so switch them back on.
# (!Nothing gets trained here - no optimizer.!Note)
for param in model.parameters():
    param.requires_grad = True

val_ds = SkinDataset("val", transform=eval_transform)


def grad_cam(image):
    # Same steps as MobileNetV2's own forward pass, done by hand so the
    # last feature map (1280 channels at 7x7) can be kept hold of.
    features = model.features(image.unsqueeze(0).to(device))
    features.retain_grad()                     # keep this tensor's gradient after backward()
    pooled = F.adaptive_avg_pool2d(features, 1).flatten(1)
    scores = model.classifier(pooled)
    pred = scores.argmax(dim=1).item()

    # gradient of the predicted class's score with respect to the feature map(see above)
    model.zero_grad()
    scores[0, pred].backward()

    # how much each channel matters = its average gradient,
    # then a weighted sum of the channels gives a 7x7 map of - "where it looked".
    importance = features.grad.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((importance * features).sum(dim=1, keepdim=True))   # keep only areas that pushed TOWARDS the prediction
    cam = F.interpolate(cam, size=(224, 224), mode="bilinear", align_corners=False).squeeze()   # stretch 7x7 up to image size
    cam = cam / (cam.max() + 1e-8)             # scale to 0-1 (1e-8 avoids dividing by zero)
    return cam.detach().cpu().numpy(), pred


sample = val_ds.df.groupby("label").sample(4, random_state=42)   # 4 images per class = 16 total

fig, axes = plt.subplots(4, 4, figsize=(12, 12))
for ax, i in zip(axes.flat, sample.index):
    row = val_ds.df.loc[i]
    image, label = val_ds[i]
    cam, pred = grad_cam(image)

    # the original photo, resized the same way with eval_transform, so the heatmap lines up
    photo = Image.open(row["path"]).convert("RGB").resize((224, 224))
    ax.imshow(photo)
    ax.imshow(cam, cmap="inferno", alpha=0.5)   # heatmap on top, half transparent so original image can still be seen, analyse the image and note differences.
    ax.set_title(f"True: {row['label']}\nPred: {CLASSES[pred]}", fontsize=9)
    ax.axis("off")

fig.tight_layout()
fig.savefig("figures/gradcam_val.png", dpi=150)
print("Saved figures/gradcam_val.png")
