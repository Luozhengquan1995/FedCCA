# FedCCA: Federated Canonical Correlation Analysis

Code release for the FedCCA paper submission (AISTATS 2026).

## Overview

FedCCA is a federated optimization framework for CCA that supports:

- centralized and federated solvers,
- privacy/noise settings,
- ablation studies on `alpha`, client number, and Neumann order.

This repository provides the exact experiment drivers used in our submission, plus a lightweight smoke test.

## Repository Structure

```text
.
├── main_all.py          # Main experiment pipeline (overall comparison)
├── main_alpha.py        # Ablation: normalization hyperparameter alpha
├── main_clientnum.py    # Ablation: number of clients
├── main_order.py        # Ablation: Neumann series order
├── main.py              # Additional sigma/order experiments
├── tals2_cca.py         # Centralized TALS-CCA
├── tals2_cca_fl.py      # Federated TALS-CCA
├── als_cca.py           # ALS baseline
├── client.py            # Federated client
├── server.py            # Federated server/aggregation
├── utils.py             # Centralized utilities
├── utils_fl.py          # Federated utilities
└── smoke_test.py        # Minimal end-to-end sanity check
```

## Environment

Tested with Python `3.10`.

Install required packages:

```bash
python3 -m pip install --user numpy scipy matplotlib
```

Note:
- Source files include `import torch`, but current NumPy/SciPy execution paths do not rely on torch APIs.
- If your environment is strict, you may either install torch or remove unused torch imports.

## Data Preparation

Create a `data/` folder and place MATLAB files:

- `data/mmill.mat`
- `data/JW11.mat`
- `data/mnist_training.mat`

Each `.mat` file should at least contain:

- `X`: first-view matrix
- `Y`: second-view matrix

If your data is `(samples, features)`, scripts automatically transpose as needed.

## Quick Start

Run a minimal sanity check first:

```bash
python3 smoke_test.py
```

Expected output includes:

- `Finish 0th iteration`, `Finish 1th iteration`, ...
- `[SMOKE] FedCCA core pipeline ran successfully.`

## Reproducing Experiments

Create output folders:

```bash
mkdir -p data result_mat \
  output/image \
  output/image_alpha \
  output/image_clientnum \
  output/image_order \
  output/image_sigma_order
```

Run experiment scripts:

```bash
python3 main_all.py
python3 main_alpha.py
python3 main_clientnum.py
python3 main_order.py
python3 main.py
```

Outputs:

- metrics in `result_mat/*.mat`
- plots in `output/image*/*.png`

## Reproducibility Notes

- Random seeds are set inside scripts (`np.random.seed(2018)` in key paths).
- For faithful reproduction, keep package versions and dataset preprocessing consistent.

## Citation

If you find this code useful, please cite our paper:

```bibtex
@inproceedings{luo2026fedcca,
  title={FedCCA: Federated Canonical Correlation Analysis},
  author={Zhengquan Luo and Kai Fong Ernest Chong and Pengfei Wei and Changyou Chen and Peilin Zhao and Renmin Han and Chunlai Zhou and Yunlong Wang and Zhiqiang Xu},
  booktitle={Proceedings of the 29th International Conference on Artificial Intelligence and Statistics (AISTATS)},
  series={Proceedings of Machine Learning Research},
  volume={300},
  year={2026}
}
```
