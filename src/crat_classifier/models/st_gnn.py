from dataclasses import dataclass

import lightning.pytorch as pl
import torch
from torch import FloatTensor, LongTensor, nn
from torch_geometric.data import Batch, Data
from torch_geometric.nn.conv import ChebConv, CGConv
from torch_geometric import nn as pyg_nn
from torch_geometric.utils import dense_to_sparse, scatter

from .base_model import BaseModelConfig


class STGraphyClassifier(pl.LightningModule):
    @dataclass
    class ModelConfig(BaseModelConfig):
        gnn_cheb_K: int = 3
        gnn_num_feats: int = 8
        gnn_hidden_channels: int = 64
        gnn_out_channels: int = 32
        gnn_reduction: str = "mean"
        rnn_layers: int = 3
        rnn_hidden_size: int = 64
        st_feats: tuple[str, ...] = ("graph", "ego_traj", "hint")

    def __init__(self, config: ModelConfig):
        super(STGraphyClassifier, self).__init__()

        self.config = config

        self.conv1 = ChebConv(
            config.gnn_num_feats,
            config.gnn_hidden_channels,
            K=config.gnn_cheb_K,
        )
        self.bn1 = pyg_nn.BatchNorm(config.gnn_hidden_channels)
        self.conv2 = ChebConv(
            config.gnn_hidden_channels,
            config.gnn_out_channels,
            K=config.gnn_cheb_K,
        )
        self.bn2 = pyg_nn.BatchNorm(config.gnn_out_channels)

        self.gru = nn.GRU(
            config.gnn_out_channels + 13,
            config.rnn_hidden_size,
            num_layers=config.rnn_layers,
            batch_first=True,
        )
        self.linear = nn.Linear(
            config.rnn_hidden_size,
            config.num_classes,
        )

    def build_agents_st_graphs(self, agent_trajs: FloatTensor, agent_type: LongTensor):
        # trajs: N * V (#agent) * 3
        # 3: [x, y, valid]
        # return T spatio-temporal describing these agents
        N, V, F = agent_trajs.shape
        # 1. build edge_index (adj-list) for full-connected graph
        # remove dumplicate edge
        adj_matrix = (torch.ones((V, V)) - torch.eye(V)).to(device=self.device)
        # 转换为edge_index
        edge_index, _ = dense_to_sparse(adj_matrix)
        # edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
        edge_index = edge_index.unsqueeze(0).repeat(N, 1, 1)

        # 2. build edge_weight with 1/distance
        dist = torch.cdist(agent_trajs[..., :2], agent_trajs[..., :2], p=2)
        edge_weight = 1.0 / dist[:, edge_index[0, 0, :], edge_index[0, 1, :]]
        edge_weight = edge_weight.clamp(1e-3, 1.0)

        # edge_weight = torch.ones(N, edge_index.shape[-1], device=self.device)
        # 3. build node features
        x = torch.cat([agent_trajs, agent_type.unsqueeze(0).repeat(N, 1, 1)], dim=-1)

        py_batched = Batch.from_data_list(
            [Data(*d) for d in zip(x, edge_index, edge_weight)]
        )
        return py_batched

    def forward(self, batch):
        batch_size = len(batch["obj_trajs"])
        batched_hint = batch["hint"]
        batched_trajs, batch_obj_types = batch["obj_trajs"], batch["obj_types"]

        batched_hint = torch.stack(batched_hint, dim=0).to(device=self.device)

        ego_trajs = torch.stack([trajs[0] for trajs in batched_trajs])
        graphy_feats = []

        batched_mask = None
        for trajs, obj_type in zip(batched_trajs, batch_obj_types):
            # iterate over batch
            graphs = self.build_agents_st_graphs(trajs.transpose(0, 1), obj_type)

            x = self.bn1(self.conv1(graphs.x, graphs.edge_index, graphs.edge_attr))  # type: ignore
            x = self.bn2(self.conv2(x, graphs.edge_index, graphs.edge_attr))  # type: ignore

            # TODO use torch.scatter
            x = scatter(x, graphs.batch, dim=0, reduce=self.config.gnn_reduction)  # type: ignore
            graphy_feats.append(x)
        graphy_feats = torch.stack(graphy_feats)  # B T F

        # st_feats = graphy_feats
        # print(batched_hint.shape, graphy_feats.shape, ego_trajs.shape)
        st_feats = torch.cat([batched_hint, graphy_feats, ego_trajs], dim=-1)
        h0 = torch.zeros(
            self.config.rnn_layers,
            batch_size,
            self.config.rnn_hidden_size,
            device=self.device,
        )
        rnn_out, hn = self.gru(st_feats, h0)
        out = self.linear(rnn_out)
        return out
        # X: N x f
        # edge_index:
