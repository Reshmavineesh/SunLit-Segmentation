#!/bin/bash

set -e

if ! command -v unzip &> /dev/null; then
    echo "[x] unzip is not installed."
    exit 0
fi

if ! command -v kaggle &> /dev/null; then
    echo "[x] kaggle is not installed, install with pip"
fi


if [ -f ~/.kaggle/kaggle.json ]; then
    echo "[+] Kaggle config available"
else
    echo "Please add Kaggle API key and re-run this script (https://github.com/Kaggle/kaggle-api#api-credentials)"
    exit 0
fi

rm -rf dataset; mkdir dataset; cd dataset

# Download from kaggle and unzip
kaggle datasets download -d pvtsec0x1/chilly-128 &
kaggle datasets download -d pvtsec0x1/chilly-160 &
wait
unzip -q chilly-128.zip -d ./dataset_chilly_128 &
unzip -q chilly-160.zip -d ./dataset_chilly_160 &
wait

# kaggle datasets download -d pvtsec0x1/sunlit-pistachio &
# kaggle datasets download -d pvtsec0x1/sunlit-tomato &
# kaggle datasets download -d pvtsec0x1/sunlit-chilly &
# kaggle datasets download -d pvtsec0x1/sunlit-chilly-128 &
# wait
# unzip -q sunlit-pistachio.zip -d ./dataset_pistachio &
# unzip -q sunlit-tomato.zip -d ./dataset_tomato &
# unzip -q sunlit-chilly.zip -d ./dataset_chilly &
# unzip -q sunlit-chilly-128.zip -d ./dataset_chilly_128 &
# wait

# kaggle datasets download -d pvtsec0x1/pistachio-leaves &
# kaggle datasets download -d pvtsec0x1/tomato-leaves &
# wait
# unzip -q pistachio-leaves.zip -d ./dataset_pistachio_128 &
# unzip -q tomato-leaves.zip -d ./dataset_tomato_128 &
# wait

rm -rf *.zip