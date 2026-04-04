# Dataset Setup — Kaggle OCT 2017

## Dataset Information

| Property       | Details                                      |
|----------------|----------------------------------------------|
| Name           | Retinal OCT Images (Kermany 2018)            |
| Classes        | CNV, DME, DRUSEN, NORMAL (4 classes)         |
| Training set   | ~83,484 images                               |
| Test set       | 968 images (242 per class, balanced)         |
| Image format   | JPEG, grayscale scanned as RGB (3-channel)   |
| Resolution     | Varies (~500–1000 px wide); resized to 224²  |
| License        | CC BY 4.0                                    |

**Citation:**
> Kermany, D. S., Goldbaum, M., Cai, W., et al. (2018). Identifying Medical
> Diagnoses and Treatable Diseases by Image-Based Deep Learning.
> *Cell*, 172(5), 1122–1131.e9.
> https://doi.org/10.1016/j.cell.2018.02.010

---

## Class Descriptions

| Class  | Disease                          | OCT signature                              |
|--------|----------------------------------|--------------------------------------------|
| CNV    | Choroidal Neovascularisation     | Subretinal fluid, membrane below RPE       |
| DME    | Diabetic Macular Edema           | Intraretinal fluid pockets, oedema         |
| DRUSEN | AMD early-stage drusen deposits  | Bumpy irregular RPE, yellow deposits       |
| NORMAL | Healthy retina                   | Smooth foveal pit, intact RPE/IS-OS layers |

**Class imbalance note:** In the training split, CNV and NORMAL are significantly
more represented than DRUSEN, which has roughly half the samples of each of the
dominant classes.  This is reflected in the per-class F1 scores — DRUSEN tends
to have the lowest recall.  Consider weighted cross-entropy or oversampling if
precision on DRUSEN matters for your application.

---

## Download Instructions

### Option A — Kaggle CLI (recommended)

```bash
# 1. Install Kaggle CLI and authenticate
pip install kaggle
# Place your kaggle.json API token at ~/.kaggle/kaggle.json

# 2. Download and unzip (from the project root)
kaggle datasets download -d paultimothymooney/kermany2018
unzip kermany2018.zip -d data/
```

### Option B — Manual Download

1. Visit: https://www.kaggle.com/datasets/paultimothymooney/kermany2018
2. Click **Download** (requires a free Kaggle account)
3. Unzip to this `data/` directory

---

## Expected Directory Structure

After downloading, your `data/` folder must look exactly like this:

```
data/
├── README.md          ← this file
├── train/
│   ├── CNV/
│   │   ├── CNV-xxxxxx.jpeg
│   │   └── ...
│   ├── DME/
│   │   └── ...
│   ├── DRUSEN/
│   │   └── ...
│   └── NORMAL/
│       └── ...
└── test/
    ├── CNV/
    │   └── ...        (242 images)
    ├── DME/
    │   └── ...        (242 images)
    ├── DRUSEN/
    │   └── ...        (242 images)
    └── NORMAL/
        └── ...        (242 images)
```

> **Note:** The Kaggle zip also contains a `val/` folder.  Our `dataset.py`
> creates its own validation split from the `train/` folder using a 90/10
> random split with fixed seed 42, so you can ignore the Kaggle `val/` folder.

---

## Quick Verification

Run this from the project root to confirm the dataset loads correctly:

```bash
python src/dataset.py data/
```

Expected output:
```
Train split — 83484 samples total:
  CNV        37206  (44.6%)  ██████████████████████████████████████████████████████████████████████████
  DME        11349  (13.6%)  ....
  DRUSEN     8617  (10.3%)  ...
  NORMAL     26312  (31.5%)  ...
...
DataLoaders ready.
Batch shape : torch.Size([32, 3, 224, 224])
```
