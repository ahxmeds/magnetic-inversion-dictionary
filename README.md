# Magnetic Inversion with Learned Dictionaries and Scale Space

This repository contains the official implementation for the paper:

**"Inversion of Magnetic Data using Learned Dictionaries and Scale Space"**  
Accepted to 10th International Conference on Scale Space and Variational Methods in Computer Vision (SSVM), 2025  
Authors: Shadab Ahamed*, Simon Ghyselincks*, Pablo Chang Huang Arias, Julian Kloiber, Yasin Ranjbar, Jingrong Tang, Niloufar Zakariaei, Eldad Haber  
[[Paper PDF]](./paper/MagneticInversionPaper.pdf) | [[arXiv]](https://arxiv.org/abs/2502.05451) 

> This codebase reproduces all experiments and results from the paper, including dataset generation, model training, and evaluation.

---

## Installation

This code is intended to be installed using Python pip. To install the package, first clone the repository and navigate to the root directory of the project where the `pyproject.toml` is found. Then, run the following command:

```bash
git clone https://github.com/ahxmeds/magnetic-inversion-dictionary.git
cd magnetic-inversion-dictionary
pip install -e .
```

This will install the `ssmaginv` package locally so that any updates to the code reflect immediately in your environment.

## Dataset

The synthetic dataset used in the paper can be regenerated using the script located at: `dataset/generate_data.py`. To reproduce the dataset used for training, validation, and testing:

```bash
python dataset/generate_data.py
```

This will create a top-level data/ directory containing:

- data/trainset/ — 2000 training samples
- data/validset/ — 100 validation samples
- data/testset/ — 500 test samples

Each .npz file in these folders contains a dictionary with:

```
save_dict = {
    "x_true": x_true_np,                     # True magnetic susceptibility model
    "data_true": data_true_np,               # Forward model output (noiseless)
    "data_true_noise": data_true_noise_np    # Noisy forward model output
}
```

Where `x_true` is the true magnetic susceptibility model, `data_true` is the noiseless forward model output, and `data_true_noise` is the noisy forward model output.

## Experiments

### Variational Method
The variational method is implemented in the `ssmaginv/variational.py` file. An grid search to determine the optimal regularization hyperparameters is provided in `experiments/variational.py`. To run a grid search, the grid values may be changed in the file and execute with:

```bash
python experiments/training/variational.py --output grid_search_variational.csv
```

The optimal values used for the experiments in the paper are:

```python
    alpha = 1e-4  # TV regularization
    beta = 1e3  # Z-weighting regularization
    z0 = 10  # Z-weighting parameter
```

Inference on the test set can be performed with:

```bash
python experiments/test_eval/evaluate_method.py --method tv
```
with an optional `--batch_size` argument to set the batch size for evaluation. The predicted models and statistics will be saved in the `RESULTS/` directory under the `Variational` folder.