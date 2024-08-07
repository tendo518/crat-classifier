# SUSCape Scene Classification

## Installation

```sh
conda create -n crat-classifier python=3.11 pip
conda activate crat-classifier

python -m venv .venv
cd PATH/TO/PROJECT
pip install -e .
```

## Usage

1. Activate conda env

    ```sh
    conda activate crat-classifier
    ```

2. train model

    ```sh
    python scripts/train.py -h  # list configuration
    python scripts/train.py
    ```
