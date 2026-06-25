# Dataset Generation Guide

To keep this repository lightweight and prevent excessive cloning times, the large 500MB **Adversarial Synthetic Fake Dataset** is strictly ignored by Git and is not hosted on GitHub.

However, all the scripts required to perfectly regenerate the dataset locally are included in this repository.

## How to Generate the Synthetic Fakes Dataset

The pipeline requires an adversarial dataset for Ablation Testing (Experiment 3). This dataset takes genuine Sri Lankan Rupee (LKR) notes and mathematically destroys exactly one specific security feature (F1 - F12) while leaving the rest of the note pristine.

To generate the full dataset of 240 synthetic fakes on your local machine, run the following command from the root directory of this repository:

```bash
python scripts/generate_synthetic_fakes.py
```

### What this script does:
1. It reads the genuine, un-augmented banknotes from `data/genuine/LKR_1000/` and `data/genuine/LKR_5000/`.
2. For each genuine note, it creates 12 mathematically altered copies.
3. It saves all 240 generated images into a newly created `data/synthetic_fakes/` directory.

Once the script finishes running (it takes a few seconds), your local dataset will be completely populated and you will be able to successfully run `python scripts/test_synthetic_fakes.py` and `python scripts/ablation_study.py`.
