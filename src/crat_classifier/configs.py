from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from crat_classifier.models import (
    CratTrajClassifier,
    EncoderDecoderClassifier,
    RNNClassifier,
)


@dataclass
class ModelConfig:
    rnn: RNNClassifier.ModelConfig
    crat: CratTrajClassifier.ModelConfig
    seq2seq: EncoderDecoderClassifier.ModelConfig
    model_name: Literal["rnn", "seq2seq", "crat"] = "rnn"


@dataclass
class OptimizerConfig:
    learning_rate: float = 5e-4
    weight_decay: float = 1e-3
    optimizer: Literal["adam", "adamw"] = "adam"
    lr_scheduler: Literal["cosine_anneal", "none"] = "cosine_anneal"


@dataclass
class ExperimentConfig:
    """experiment configuration"""

    train_split: Path = Path("data/suscape/train")
    # training split
    val_split: Path = Path("data/suscape/val")
    # val split
    output_root: Path = Path("output")
    # use early stop
    early_stop: bool = False
    # output root dir
    num_epochs: int = 5000
    # training epochs
    seed: int | None = None
    # random seed, None for not set
    batch_size: int = 64
    # learning rate
    num_workers: int = 8
    # workders for dataloder
    gpus: int = 1
    # gpu to use


@dataclass
class Config:
    """overall config"""

    model: ModelConfig
    # model configuration
    optimizer: OptimizerConfig
    # optimizer configuration
    experiment: ExperimentConfig
    # experiment configuration
