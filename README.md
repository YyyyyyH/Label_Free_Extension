# Label-Free Explainability Extension for Unsupervised Models

This repository is an extension of the work presented in "Label-Free Explainability for Unsupervised Models". The original paper can be found [here](https://arxiv.org/abs/2203.01928). This repository's comprehensive findings and detailed analyses are encapsulated in a singular report, located in the ./Report directory for convenient access and reference.

## Installation

Set up your environment to run the extension code by following these instructions:

```bash
conda env create -f environment.yml
conda activate LFE
```

This will create a new Conda environment named 'LFE' with all the necessary dependencies.

## Usage

The repository's structure is primarily divided into two main sections, all located within the `/src` directory.

### Part 1: Medical Image Testing

This section is focused on applying the concepts to various medical image datasets. To get started, you can run the following scripts:

- `BreastMNIST.py` for Breast MNIST dataset analysis.
- `ChestMNIST.py` for Chest MNIST dataset analysis.
- `OrganSMNIST.py` for Organ-Specific MNIST dataset analysis.

To execute any of these scripts, navigate to the `/src` directory and run:

```bash
python ./src/<filename>.py
```

Replace `<filename>` with the respective script's name.

### Part 2: FactorVAE Experiment

The FactorVAE experiment can be initiated with the command below:

```bash
python ./src/dsprites.py
```

This script conducts experiments using the FactorVAE model on the dSprites dataset.

### General Instructions for Running Scripts

- Ensure that you are in the root directory of the repository before executing any scripts.
- All output files, including logs and results, will be automatically saved to the `../results` directory for easy access and review.

