# Label-Free Explainability Extension for Unsupervised Models

This repository is an extension of the work presented in "Label-Free Explainability for Unsupervised Models". The original paper can be found [here](https://arxiv.org/abs/2203.01928).

## 1. Installation

To set up the environment for running the extension code, please follow these steps:

```bash
conda env create -f environment.yml
conda activate LFE
```

## 2. Usage

The main code for this project is stored in the `/src` directory. This project is divided into two primary parts:

### Part 1: Medical Image Testing

This includes tests on different types of medical images, such as:

- `BreastMNIST.py`
- `ChestMNIST.py`
- `OrganSMNIST.py`
  
To run these files, use:
```bash
python ./src/<filename>.py
```
### Part 2: FactorVAE Experiment

To run the FactorVAE experiments, use the following command:

```bash
python ./src/dsprites.py
```

### Running the Code

Please ensure that you are in the root repository when executing these scripts.
