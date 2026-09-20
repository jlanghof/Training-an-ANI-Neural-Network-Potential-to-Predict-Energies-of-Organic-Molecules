# ANI Neural Network Potential for Molecular Energy Prediction

Final project for **CHEM C142/242: Machine Learning, Statistical Models, and Optimization for Molecular Problems** at UC Berkeley (Spring 2026).

In this project I built and trained an ANI-style neural network potential that predicts the energy of small organic molecules from their 3D atomic coordinates. My final model reaches a **test MAE of 1.58 kcal/mol**, which beats the project's 2 kcal/mol target and lands close to the 1.3 kcal/mol RMSE reported in the original ANI-1 paper, while training on about 4% as much data.

![Parity plot](figures/parity_final.png)

## How it works

ANI models don't look at a molecule as a whole. Instead, each atom gets described by an **Atomic Environment Vector (AEV)**, a fixed-length fingerprint of the atoms around it built from radial and angular symmetry functions. Each element (H, C, N, O) has its own small neural network that takes an atom's AEV and outputs that atom's energy contribution. The molecular energy is the sum over all atoms.

```
coordinates + species ──► AEV computer (384-dim per atom) ──► net_H / net_C / net_N / net_O ──► sum ──► molecular energy
```

This makes the model invariant to rotation, translation, and atom ordering, and lets it handle molecules of different sizes.

## Data

I used the `ani_gdb_s01_to_s04.h5` subset of the ANI-1 dataset: DFT energies for off-equilibrium conformations of molecules with 1 to 4 heavy atoms (C, N, O). Before training I subtracted per-element self-energies so the network learns the interaction energy instead of the huge constant atomic terms, then did an 80/10/10 train/validation/test split with a batch size of 8192.

The dataset file is not included in this repo because of its size. It's part of the public ANI-1 release (see References).

## Model

| Setting | Value |
|---|---|
| AEV | Radial cutoff 5.2 Å, angular cutoff 3.5 Å, 384 features per atom |
| Atomic network | 384 → 256 → 128 → 1, ReLU (one network per element) |
| Parameters | 526,340 |
| Optimizer | Adam, lr = 1e-3, weight decay (L2) = 1e-5 |
| Training | 20 epochs, MSE loss, early stopping on validation loss |

## Results

### Hyperparameter search

I started from a 1-hidden-layer baseline (2.13 kcal/mol) and tested six configurations:

| Trial | Change | Test MAE (kcal/mol) |
|---|---|---|
| 1 | Baseline, 20 epochs | 1.97 |
| 2 | **Extra hidden layer (384→256→128→1)** | **1.70** |
| 3 | Lower learning rate (5e-4), 30 epochs | 3.89 |
| 4 | Stronger L2 (1e-4) | 2.79 |
| 5 | Dropout (p = 0.1) | 4.67 |
| 6 | Three hidden layers | 3.62 |

The deeper 2-hidden-layer network won. Dropout, stronger weight decay, and a third hidden layer all hurt, which tells me the model was underfitting more than overfitting at this dataset size.

### Robustness checks

- **Repeated runs:** mean MAE of 1.96 kcal/mol (std 0.34), with the spread coming from random initialization
- **3-fold cross-validation:** mean MAE of 1.98 kcal/mol (std 0.18)

### Final model

| Heavy atoms | Molecules in test set | MAE (kcal/mol) |
|---|---|---|
| 1 | 1,052 | 1.59 |
| 2 | 5,087 | 3.07 |
| 3 | 15,176 | 1.56 |
| 4 | 65,174 | 1.47 |
| **Overall** | **86,489** | **1.58** |

2-heavy-atom molecules are the one weak spot. They're a small and less diverse slice of the data, so the model has less to generalize from there.

### Comparison to the ANI-1 paper

| | ANI-1 paper | This project |
|---|---|---|
| Training conformations | ~17 million | ~692k |
| Heavy atoms covered | 1–8 | 1–4 |
| Architecture | 768:128:128:64:1 | 384:256:128:1 |
| Activation | Gaussian | ReLU |
| Test error | 1.3 kcal/mol (RMSE) | 1.58 kcal/mol (MAE) |

## Repo structure

```
├── project_notebook.ipynb   # Full notebook, checkpoints 1–5 with dated progress notes
├── report.pdf               # Final written report
├── figures/                 # Learning curves, parity plots, heavy-atom breakdown
└── README.md
```

## Running it

```bash
pip install torch torchani numpy matplotlib tqdm h5py
```

1. Download `ani_gdb_s01_to_s04.h5` from the ANI-1 dataset and put it in the repo root.
2. Open `project_notebook.ipynb` and run the cells in order.

A GPU is strongly recommended. I trained on UC Berkeley's Savio cluster (GPU partition), where one 20-epoch run took about 2 minutes. The notebook falls back to CPU automatically, but training will be much slower.

## Limitations

- The model only saw molecules with up to 4 heavy atoms, so I wouldn't trust it on anything larger.
- It only predicts energies, not forces.
- It only covers H, C, N, and O.

## References

- J. S. Smith, O. Isayev, A. E. Roitberg. *ANI-1: an extensible neural network potential with DFT accuracy at force field computational cost.* Chemical Science 8, 3192 (2017).
- J. S. Smith, O. Isayev, A. E. Roitberg. *ANI-1, A data set of 20 million calculated off-equilibrium conformations for organic molecules.* Scientific Data 4, 170193 (2017).
- [TorchANI](https://github.com/aiqm/torchani) and its [training example](https://aiqm.github.io/torchani/examples/nnp_training.html), which the notebook structure is based on.

## Author

Jack Langhoff, UC Berkeley
