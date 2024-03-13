#!/bin/bash

# Download the dataset from Google Drive
wget --no-check-certificate 'https://drive.google.com/uc?export=download&id=1ZeX-M2fA-HKNg1nWwDDsI66O6seUwpz4' -O 'CAMELS.zip'

# Unzip the downloaded dataset into the specified directory
unzip 'CAMELS.zip' -d 'data/'

# Remove the downloaded zip file
rm 'CAMELS.zip'

echo "Dataset downloaded and extracted successfully."
