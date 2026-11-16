#!/usr/bin/env bash
# Downloads PlantVillage (lab photos) and PlantDoc (field photos) into data/raw/.
# PlantVillage is large (several GB); this takes a while.
set -euo pipefail

mkdir -p data/raw
cd data/raw

if [ ! -d PlantDoc-Dataset ]; then
  echo "Downloading PlantDoc (field photos)..."
  git clone --depth 1 https://github.com/pratikkayal/PlantDoc-Dataset.git
fi

if [ ! -d PlantVillage-Dataset ]; then
  echo "Downloading PlantVillage (lab photos)..."
  git clone --depth 1 https://github.com/spMohanty/PlantVillage-Dataset.git
fi

cd ../..
echo
echo "Done. Next:"
echo "  python scripts/prepare_data.py --plantvillage data/raw/PlantVillage-Dataset/raw/color --plantdoc data/raw/PlantDoc-Dataset"
echo
echo "If the PlantVillage clone is slow, use the Kaggle copy instead (see data/README.md)."
