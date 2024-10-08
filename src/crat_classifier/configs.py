from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from crat_classifier.models import (
    RNNClassifier,
    STGraphyClassifier,
)


@dataclass
class ModelConfig:
    rnn: RNNClassifier.ModelConfig
    # crat: CratTrajClassifier.ModelConfig
    # seq2seq: EncoderDecoderClassifier.ModelConfig
    st_gnn: STGraphyClassifier.ModelConfig
    model_name: Literal["rnn", "st_gnn"] = "st_gnn"


@dataclass
class OptimizerConfig:
    learning_rate: float = 1e-4
    weight_decay: float = 1e-3
    optimizer: Literal["adam", "adamw"] = "adam"
    lr_scheduler: Literal["cosine_anneal", "none"] = "none"


@dataclass
class ExperimentConfig:
    """experiment configuration"""

    train_split: Path = Path("data/suscape_trajs/train")
    # training split
    val_split: Path = Path("data/suscape_trajs/val")
    # val split
    output_root: Path = Path("output")
    # use early stop
    early_stop: bool = False
    # output root dir
    num_epochs: int = 1500
    # training epochs
    seed: int | None = None
    # random seed, None for not set
    batch_size: int = 32
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
