# PINNs-TF2.7-light-R4

Physics-informed neural networks (PINNs) for reconstructing pulsatile flow fields in straight and curved pipes from sparse velocity data.

This repository accompanies the manuscript:

> **Reconstruction of pulsatile flow in curved pipes using physics-informed neural networks**
> Xingchao Zhang, Fan Wu, Yunfan Yang, Hongping Wang, Shizhao Wang
> Submitted to *European Journal of Mechanics - B/Fluids* (manuscript no. EJMFLU-D-26-00743).

The PINN embeds the incompressible Navier–Stokes equations into the loss function and reconstructs the velocity and pressure fields simultaneously from velocity measurements (e.g., PIV-type data).

## Reproducible example

The data and weights included here correspond to the **single-plane (single-cross-section) configuration** of the manuscript: a three-dimensional curved pipe at Re = 1000, Wo = 10, β = π/32, in which the PINN is trained only on the streamwise (z = 0) measurement plane, so that the spatial-coverage effect of single-plane training data on pressure reconstruction can be reproduced and compared.

| File | Description |
|---|---|
| `data/bendpipe/2d3c_Wo10Bpi32_reslu40_noise0_pinn.mat` | Training data for the single-plane case (3D curved pipe, Re = 1000, Wo = 10, β = π/32; velocity data on the z = 0 plane) |
| `weights/2d3c_Wo10Bpi32_13_156_run0/` | Trained PINN weights for this single-plane case |

Note: the pre-computed prediction file for this case (z = 0 plane) exceeds the GitHub file-size limit and is therefore not included; it can be regenerated from the provided weights with the instructions below.

## Directory structure

```
.
├── train.py                     # Training entry (contains the 3D curved-pipe case)
├── predict.py                   # Prediction entry (loads trained weights, predicts on planes)
├── datagenerator.py             # Data loading / PIV-type data generation
├── pinns_3d.py                  # 3D PINN model
├── funcs.py                     # Helper functions (LevenbergMarquardt, generate_dataset)
├── maps.py                      # Network architecture (ResNet)
├── userbackend.py               # TensorFlow backend / GPU configuration
├── autograd-minimize/           # Third-party dependency (required by pinns_3d)
├── data/
│   └── bendpipe/
│       └── 2d3c_Wo10Bpi32_reslu40_noise0_pinn.mat   # Single-plane training data (3D curved pipe, Re=1000, Wo=10, β=π/32)
└── weights/
    └── 2d3c_Wo10Bpi32_13_156_run0/   # Trained weights for the single-plane case
```

## Environment

The code was developed and tested with:

- Python 3.9
- TensorFlow 2.7 (GPU)
- CUDA 11.x / cuDNN 8.2
- numpy, scipy, h5py, matplotlib, sympy

Install the required packages:

```bash
pip install numpy scipy h5py matplotlib sympy tensorflow==2.7
```

The repository vendors the `autograd-minimize` package (used by `pinns_3d.py` for BFGS-type fine-tuning). It is imported directly from this directory, so no separate installation is needed. If you prefer to install it from PyPI instead, run:

```bash
pip install autograd-minimize
```

and remove the vendored `autograd-minimize/` folder (or keep it; the import will use the installed package).

## Usage

1. **Generate predictions (with the provided weights)**

   Run `python predict.py` (or execute `predict_3d3c_AOCFD()`). The script reads the mesh and time range from `data/bendpipe/2d3c_Wo10Bpi32_reslu40_noise0_pinn.mat`, loads the trained weights `weights/2d3c_Wo10Bpi32_13_156_run0`, predicts the velocity and pressure fields on the z = 0 plane over one pulsation period, and saves the result to `predict_results/2d3c_Wo10Bpi32_13_156_run0_plane_z0_predict.mat`.

2. **Retrain the model from scratch**

   The corresponding training setup is described in `train.py` (three-dimensional curved-pipe case, 13 hidden layers × 156 neurons, adaptive loss weighting, Adam + optional L-BFGS). Run `python train.py`; the trained weights are saved under the same `savename` (`2d3c_Wo10Bpi32_13_156_run0`), so `predict.py` can then be run directly to regenerate the predictions. Training a 3D case requires a GPU and takes on the order of 10–15 hours.

## Reproducing the parameter sweep / data-coverage studies

The manuscript also reports a Womersley-number / amplitude-ratio parameter sweep (`Wo = 0–15`, `β = π/64–π/8`) and a comparison of four training-data compositions (single-plane, assumed-parabolic inlet, orthogonal cross-plane, and full 3D). The single-plane curved-pipe case included here is representative of this data-coverage study. The training/prediction routines for the additional cases (other `Wo`/`β` combinations and other data compositions) follow the same code structure as the provided single-plane example, by switching the data file and `savename` in `train.py` / `predict.py`; the associated CFD data files for those cases are large (>100 MB) and are not included in this repository. They can be requested from the corresponding author.

## Citation

If you use this code, please cite the manuscript above (once published) and the PINN references it builds upon (Raissi et al., 2019; Jin et al., 2021; Jagtap & Karniadakis, 2020).
