# SUSCape Scene Classification

## Installation

```sh
conda create -n crat-classifier python=3.12 pip
conda activate crat-classifier

cd PATH/TO/PROJECT
pip install -e .
```

## Usage

1. Activate venv

    ```sh
    source .venv/bin/activate
    ```

2. train model

    ```sh
    python scripts/train.py -h  # list configuration
    python scripts/train.py
    ```
