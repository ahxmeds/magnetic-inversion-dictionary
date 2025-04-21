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

This will install the `ssmaginv` package locally so that any updates to the code reflect immediately in your environment. Most project dependencies are included in the `pyproject.toml` file. A `requirements.txt` and `environment.yml` file are also provided for convenience. 

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

### Cosine Dictionary
**TODO**: Add cosine dictionary training script

**Inference** on the test set can be performed with:

```bash
python experiments/test_eval/evaluate_method.py --method cosine
```

### Shared Dictionary
Training can be done using the general dictionary training script located at `experiments/training/dictionary.py`. The training script takes the following arguments:
```bash
python experiments\training\dictionary.py --model single --bs 16 --epochs 200 
```
where `--model` is set to be a single shared dictionary, `--bs` is the batch size, and `--epochs` is the number of epochs to train. The training script will save the best model weights in the `RESULTS/WEIGHTS` directory. The training script will also save the training and validation loss in a CSV file in logs folder.

**Inference** on the test set can be performed with:

```bash
python experiments/test_eval/evaluate_method.py --method single_dict
```

with predictions stored in `RESULTS/PREDS` in the `SingleDict` folder.

### Learned Dictionary
Training can be done using the general dictionary training script located at `experiments/training/dictionary.py`. The training script takes the following arguments:
```bash
python experiments\training\dictionary.py --model scale --bs 16 --epochs 200 
```
where `--model` is set to be a single shared dictionary, `--bs` is the batch size, and `--epochs` is the number of epochs to train. The training script will save the best model weights in the `RESULTS/WEIGHTS` directory. The training script will also save the training and validation loss in a CSV file in logs folder.

**Inference** on the test set can be performed with:

```bash
python experiments/test_eval/evaluate_method.py --method shared_dict
```

with predictions stored in `RESULTS/PREDS` in the `UnrolledDict` folder.

---
## Images and Analysis

#### Compare Methods Visually
Once predicted models are computed over the test set, the results can be visualized and/or saved with a helper plotting script. Use the `--save` flag to change between viewing and saving. 

```bash
python experiments/plotting/compare_3dmodels_plot.py --index i

## or
python experiments/plotting/compare_3dmodels_plot.py --index i --save
```
where `i` is the index of the sample to visualize.

The test results across all methods for samples `1,11,19` have been saved in the figures directory.

#### Compare Methods Statistics

The statistics of the predicted models can be computed and saved with the following command:

```bash
python experiments/plotting/loss_comparison_plot.py
```

The results for from the paper are saved in the `figures/` directory. 

