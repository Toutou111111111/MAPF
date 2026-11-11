# MAPF

Code for **MAPF: Multi-Path Adaptive Forecasting for Long-Term Time Series**.

## Model Implementation

MAPF uses three forecasting paths for slow trends, recurring patterns, and local variations. The trend path combines smoothing scales adaptively. The harmonic and wavelet-detail paths operate in parallel on the same detrended history. Learned gates combine the three forecasts.

The model is implemented in [`models/MAPF.py`](models/MAPF.py), with data loading in [`data_provider/data_loader.py`](data_provider/data_loader.py) and training and evaluation in [`exp/exp_main.py`](exp/exp_main.py).

## Getting Started

### Environment

The experiments used Python 3.10.19, PyTorch 2.9.1 with CUDA 12.8, and an NVIDIA A40 GPU.

```bash
conda create -n mapf python=3.10.19 -y
conda activate mapf
python -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
```

### Data Preparation

The experiments cover ETTh1, ETTh2, ETTm1, ETTm2, Electricity, Traffic, and Weather. Benchmark data are available from [ETDataset](https://github.com/zhouhaoyi/ETDataset) and the [Time-Series-Library data resources](https://github.com/thuml/Time-Series-Library#prepare-data).

Place the CSV files under `./data/`, matching `--root_path` and `--data_path` in each experiment script. For example, the ETTh1 script expects:

```text
./data/ETT/ETTh1.csv
```

### Running Experiments

Run the dataset scripts from the repository root. For instance:

```bash
bash scripts/MAPF/etth1.sh
nohup bash scripts/MAPF/etth1.sh > etth1.log 2>&1 &
bash scripts/MAPF/weather.sh
nohup bash scripts/MAPF/weather.sh > weather.log 2>&1 &
```

This script runs prediction lengths of 96, 192, 336, and 720 with an input length of 512. Use the settings in each script to reproduce the corresponding experiment.

Checkpoints are saved under `./checkpoints/`, and test metrics under `./results/`. MSE and MAE are also appended to `result.txt`. Test errors are computed on the standardized data scale.
