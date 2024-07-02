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
        bidirection: bool = True

    def __init__(self, config: ModelConfig):
        super(RNNClassifier, self).__init__()

        self.config = config

        self.rnn_D = 2 if config.bidirection else 1
        self.rnn = nn.GRU(
            9,
            hidden_size=config.hidden_sizes,
            num_layers=config.num_layers,
            batch_first=True,
            bidirectional=config.bidirection,
            dropout=config.dropout_ratio,
        )
        self.linear = nn.Linear(config.hidden_sizes * self.rnn_D, config.num_classes)

    def forward(self, batch):
        displ, centers = batch["displ"], batch["centers"]
        hint = batch["hint"]
        agents_per_sample = [x.shape[0] for x in displ]

        hint = torch.stack(hint, dim=0).to(device=self.device)
        ego_displ = torch.stack([x[0] for x in displ], dim=0).to(device=self.device)
        
        ego_displ_forward = torch.concat(
            (
                torch.zeros((ego_displ.shape[0], 1, 4), device=self.device),
                ego_displ,
            ),
            dim=1,
        )
        ego_displ_backward = torch.concat(
            (
                -ego_displ,
                torch.zeros((ego_displ.shape[0], 1, 4), device=self.device),
            ),
            dim=1,
        )

        feat_in = torch.concat((ego_displ_forward, ego_displ_backward), dim=-1)
        feat_in = torch.concat((feat_in, hint.unsqueeze(-1)), dim=-1)
        h0 = torch.randn(
            (
                self.rnn_D * self.config.num_layers,
                feat_in.shape[0],
                self.config.hidden_sizes,
            ),
            device=self.device,
        )
        rnn_out, hn = self.rnn(feat_in, h0)
        out = self.linear(rnn_out)
        return out
