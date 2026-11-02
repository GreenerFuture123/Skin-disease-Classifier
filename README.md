[README.md](https://github.com/user-attachments/files/34280092/README.md)
# Skin Condition Classification with MobileNetV2

A PyTorch transfer-learning project that classifies images into four dataset labels: Chickenpox, Measles, Monkeypox and Normal. MobileNetV2 achieved a test macro-F1 of **0.85 ± 0.04** and accuracy of **87% ± 3 percentage points**, averaged across three training seeds.

> **Disclaimer:** This is a learning and portfolio project. It is not a medical tool and must not be used for diagnosis.

## Dataset

This project uses the [Infectious Skin Disease Dataset on Kaggle](https://www.kaggle.com/datasets/gupiii/infectious-skin-disease-dataset), licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The dataset's citation is:

> Prayoga, L. F. (2026). *Infectious Skin Disease Dataset & Classification using Hybrid MobileNetV2-Xception with CLAHE Image Enhancement* (Undergraduate thesis, Universitas Airlangga). GitHub Repository: https://github.com/lutfi015/skin-disease

Only `Dataset_Gabungan` was used. It combines `Dataset_1` and `Dataset_2`, so adding those folders again would count images twice.

| Class | Original images | After exclusions |
|---|---:|---:|
| Chickenpox | 207 | 205 |
| Measles | 171 | 171 |
| Monkeypox | 543 | 541 |
| Normal | 508 | 508 |
| **Total** | **1,429** | **1,425** |

Changes made for this project: four files with conflicting labels were excluded, and duplicate or near-duplicate images were grouped before splitting the data. Remaining duplicates were retained within their assigned split.

Dataset images are not included in this repository. Download them from the Kaggle link above.

## Data problems and what I did

### Duplicates

Perceptual hashing (pHash) found **624 image pairs with identical hashes**, including **598 pairs between the two source datasets**. Inspection showed that `Dataset_2` largely contains images from `Dataset_1` saved in different formats and sizes.

A random image-level split could therefore place copies of the same photograph in training and test data, inflating performance.

I grouped images with a pHash Hamming distance of **10 or less**, then kept every group entirely within one split. I visually checked all **17 pairs** at distances 6–10 or with mismatched labels.

The deliberately loose threshold favours keeping potential copies together. A false match can reduce the effective sample size or affect split balance, but it does not place those matched images across splits. A missed duplicate can still cause leakage.

### Label conflicts

Two photographs appeared under both Chickenpox and Monkeypox labels, across four files:

- `ds1_ChickenPox_88` and `ds1_MonkeyPox_82`
- `ds2_chickenpox94` and `ds2_monkeypox84`

I excluded all four files because I could not establish the correct labels. This left **1,425 images**.

### Class imbalance

Chickenpox and Measles each have roughly one-third as many images as the larger classes. Accuracy alone could therefore hide weak performance on these classes.

Training used class weights, and evaluation reports macro-F1, per-class precision and recall, and a confusion matrix.

## Method

### Split

I used `StratifiedGroupKFold` with **seven folds** and `random_state=42`: one fold for test, one for validation and five for training. The fixed assignments were saved in `data/splits.csv` and reused across experiments.

| Split | Images | Photo groups | Chickenpox | Measles | Monkeypox | Normal |
|---|---:|---:|---:|---:|---:|---:|
| Training | 1,017 | 533 | 146 | 121 | 386 | 364 |
| Validation | 204 | 106 | 30 | 25 | 77 | 72 |
| Test | 204 | 105 | 29 | 25 | 78 | 72 |

Checks confirmed that no duplicate group crossed split boundaries and that the four conflicting files were excluded.

The test set was evaluated once for each of the three final models, after the setup was locked. No tuning or further training runs followed inspection of the test results.

### Image processing

All images were resized to **224 × 224**, converted to tensors and normalised using the ImageNet mean and standard deviation expected by the pretrained model.

Training-only augmentation used:

- Random horizontal flips
- Random rotations up to 15°
- Brightness adjustment of 0.2

Hue and saturation were preserved because colour may help distinguish the dataset classes.

### Training

The model was MobileNetV2 with ImageNet pretrained weights and a replacement four-class output layer. Both phases used class weights and selected checkpoints by **lowest validation loss**.

1. **Frozen phase:** train the classifier for 10 epochs with the base frozen, including its BatchNorm layers, using Adam at a learning rate of `1e-3`.
2. **Fine-tuning phase:** start from the selected frozen checkpoint and fine-tune the last four base blocks for 10 epochs at a learning rate of `1e-4`.

Batch size was **32**. Seeds **42, 1 and 2** were chosen in advance, and every seed used the same data split. The final setup did not use CLAHE.

## Results

Values below are **mean ± standard deviation across three seeds**. This variation reflects training randomness on one fixed split; it is not a confidence interval for performance on new populations.

| Metric | Test | Validation |
|---|---:|---:|
| Accuracy | 87.3% ± 2.6 pp | 91.5% ± 0.6 pp |
| Macro-F1 by image | 0.85 ± 0.04 | 0.88 ± 0.01 |
| Macro-F1 by unique photo | 0.84 ± 0.03 | 0.88 ± 0.02 |

Photo-level results help account for repeated copies of the same photograph. Photo groups are inferred from image similarity and do not represent verified patient identities.

### Per-class test results

Precision, recall and F1 are averaged across seeds. Image counts describe the fixed test set.

| Class | Precision | Recall | F1 | Images |
|---|---:|---:|---:|---:|
| Chickenpox | 0.68 | 0.75 | 0.71 | 29 |
| Measles | 0.79 | 0.96 | 0.86 | 25 |
| Monkeypox | 0.95 | 0.78 | 0.86 | 78 |
| Normal | 0.92 | 0.99 | 0.96 | 72 |

Always predicting **Monkeypox**, the largest training class, gives **38.2% test accuracy** (78/204) and image-level macro-F1 of approximately **0.14**. On validation, the same baseline gives 37.7% accuracy.

![Confusion matrix of test-set predictions across three seeds](figures/test_confusion_matrix.png)

Full evaluation output is recorded in `results/test_results.txt`.

## Findings

**Fine-tuning improved validation macro-F1 on all three seeds.** Mean macro-F1 rose from approximately 0.83 with the frozen base to 0.88 after fine-tuning.

**CLAHE showed no benefit on this validation set.** Mean validation macro-F1 was 0.87 with CLAHE and 0.88 without it. The rule, set before seeing results, was to adopt CLAHE only if it beat no-CLAHE by more than the seed variation (about 0.02). It did not, so it was excluded from the final setup. This does not establish that CLAHE is generally harmful.

**Chickenpox remained the weakest class.** Its mean test F1 was 0.71, and recall varied from 0.62 to 0.90 across seeds.

**Monkeypox was frequently confused with Chickenpox.** Each of the three final models classified 10 of the 78 Monkeypox test images as Chickenpox. Mean Monkeypox recall fell from approximately 0.90 on validation to 0.78 on test.

**Test scores were lower than validation scores.** This is consistent with making several development choices using validation performance. The small effective sample also makes differences between splits plausible.

## Limitations

- **Small effective sample:** the test set contains 204 images but only 105 inferred photo groups. One group holds two labels because of a false match, so unique-photo results count 106 photos. Repeated images are not independent observations.
- **No patient IDs:** different photographs of the same patient could remain in different splits.
- **Possible shortcuts (seen by eye in a sample, not measured):** some images contain watermarks or publication artefacts. Normal images often resemble stock or skincare-advert photographs, while Chickenpox images mostly appear to show children.
- **Limited interpretability checks:** a visual Grad-CAM review covered only 16 validation images from one model. Attention on skin does not rule out shortcuts.
- **Fairness not evaluated:** performance across skin tones was not measured. Apparent differences in skin-tone representation between classes could confound predictions.
- **Validation reuse:** repeated comparisons may favour choices that fit this particular validation set.
- **No clinical validation:** these dataset results do not establish performance in clinical practice.

## How to run

Requires Python 3.13. Create and activate a virtual environment, then install the dependencies:

```bash
pip install -r requirements.txt
```

Download the dataset and place `Dataset_Gabungan` at `data/raw/Dataset_Gabungan/`.

Run the pipeline from the repository root in this order:

```bash
python src/duplicates.py     # group duplicate and near-duplicate images
python src/make_split.py     # exclude the conflicting files and create the fixed grouped split
python src/train.py          # train both phases (see below)
python src/final_test.py     # evaluate the three selected models on the test set
python src/plot_results.py   # make the confusion matrix from numbers recorded in the script
```

`train.py` has no command-line options. Set `SEED` at the top of the file and run it once each for **42**, **1** and **2**, with `USE_CLAHE = False`. Models are saved in the repository root and are not included in this repository. Results may differ slightly on other machines.

Two optional scripts produced the validation findings: `src/gradcam.py` (Grad-CAM heatmaps) and `src/errors.py` (images the models repeatedly get wrong).
