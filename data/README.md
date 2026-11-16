# Data

| Dataset | Domain | What it is | Source |
|---|---|---|---|
| PlantVillage | lab | ~54k leaf photos on plain backgrounds, 38 classes | github.com/spMohanty/PlantVillage-Dataset (or Kaggle: "plantvillage dataset") |
| PlantDoc | field | ~2.6k real-world photos from the internet, 27 classes, cluttered backgrounds | github.com/pratikkayal/PlantDoc-Dataset |

Only the **28 classes present in both** are used (`configs/label_map.yaml`), so the
lab → field comparison is fair.

## Steps

```bash
bash scripts/download_data.sh
python scripts/prepare_data.py \
    --plantvillage data/raw/PlantVillage-Dataset/raw/color \
    --plantdoc data/raw/PlantDoc-Dataset
```

If you use the Kaggle copy of PlantVillage, point `--plantvillage` at its `color/`
folder (the one that directly contains the class folders).

`prepare_data.py` prints images per class for each domain and warns about any
PlantDoc folder it could not map. Folder names differ slightly between dataset
versions; add variants to `label_map.yaml` if needed.

## Things to mention in your report

- PlantVillage has several photos of the same leaf, so a random split can leak
  near-duplicates between train and test and inflate lab accuracy. This makes the
  lab → field gap look even larger. A stricter split is a good extension.
- PlantDoc is small and some classes have very few test images. Report per-class
  numbers and macro-F1, not only accuracy.
- **Best extension:** collect 200–500 of your own phone photos from local farms or
  a college garden, labelled with help from an agriculture expert. A small "Indian
  field test set" is a strong, original contribution.
