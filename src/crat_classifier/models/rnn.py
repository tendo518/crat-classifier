from dataclasses import dataclass

import lightning.pytorch as pl
import torch
from torch import nn

from .base_model import BaseModelConfig


class RNNClassifier(pl.LightningModule):
    @dataclass
    class ModelConfig(BaseModelConfig):
        num_layers: int = 2
        hidden_sizes: int = 32
        dropout_ratio: float = 0.3
        bidirection: bool = False

    def __init__(self, config: ModelConfig):
        super(RNNClassifier, self).__init__()

        self.config = config

        self.rnn_D = 2 if config.bidirection else 1
        self.rnn = nn.GRU(
            4,
            hidden_size=config.hidden_sizes,
            num_layers=config.num_layers,
            batch_first=True,
            bidirectional=config.bidirection,
            dropout=config.dropout_ratio,
        )
        self.linear = nn.Linear(config.hidden_sizes * self.rnn_D, config.num_classes)

    def forward(self, batch):
        displ, centers = batch["displ"], batch["centers"]
        agents_per_sample = [x.shape[0] for x in displ]

        ego_displ = torch.stack([x[0] for x in displ], dim=0)
        # N 39 3 ego car displ
        ego_displ = torch.concat(
            (torch.zeros((ego_displ.shape[0], 1, 4), device=self.device), ego_displ),
            dim=1,
        )
        h0 = torch.randn(
            (
                self.rnn_D * self.config.num_layers,
                ego_displ.shape[0],
                self.config.hidden_sizes,
            ),
            device=self.device,
        )
        # (N*agents_per_sample) 39 latent_size
        rnn_out, hn = self.rnn(ego_displ, h0)
        # (N*agents_per_sample) 39 num_classes
        out = self.linear(rnn_out)
        return out
