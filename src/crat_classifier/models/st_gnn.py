from dataclasses import dataclass

import lightning.pytorch as pl
import torch
from torch import FloatTensor, LongTensor, nn
from torch_geometric import nn as pyg_nn
from torch_geometric.data import Batch, Data
from torch_geometric.nn.conv import ChebConv, CGConv, GATConv, GCNConv
from torch_geometric.utils import dense_to_sparse, scatter
from crat_classifier.utils import unique_with_index
from .base_model import BaseModelConfig


class STGraphyClassifier(pl.LightningModule):
    @dataclass
    class ModelConfig(BaseModelConfig):
        gnn_cheb_K: int = 3
        gnn_num_feats: int = 14
        gnn_hidden_channels: int = 128
        gnn_out_channels: int = 64
        gnn_reduction: str = "ego_only"
        gnn_dp_ratio: float = 0.2
        rnn_layers: int = 2
        rnn_hidden_size: int = 32
        st_feats: tuple[str, ...] = ("graph", "ego", "hint")

    def __init__(self, config: ModelConfig):
        super(STGraphyClassifier, self).__init__()

        self.config = config

        self.conv1 = GATConv(
            config.gnn_num_feats,
            config.gnn_hidden_channels,
            dropout=config.gnn_dp_ratio
            # K=config.gnn_cheb_K,
        )
        self.conv2 = GATConv(
            config.gnn_hidden_channels,
            config.gnn_out_channels,
            dropout=config.gnn_dp_ratio
            # K=config.gnn_cheb_K,
        )
        # self.cgconv = GATConv(config.gnn_num_feats, config.gnn_hidden_channels, dropout=config.gnn_dp_ratio)
        self.ego_gru = nn.GRU(
            3, config.rnn_hidden_size, num_layers=config.rnn_layers, batch_first=True
        )
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

    def build_agents_st_graphs(
        self, agent_trajs: FloatTensor, agent_type: LongTensor, agent_vels: FloatTensor
    ):
        # trajs: N * V (#agent) * 3
        # 3: [x, y, valid]
        # return T spatio-temporal describing these agents
        N, V, F = agent_trajs.shape
        # 1. build edge_index (adj-list) for full-connected graph
        # remove dumplicate edge
        adj_matrix = (torch.ones((V, V)) - torch.eye(V)).to(device=self.device)
        # OR: use edge with ego only
        # adj_matrix = torch.zeros((V, V)).to(device=self.device)
        # adj_matrix[0, :] = 1.0
        # adj_matrix[:, 0] = 1.0
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

        # [sin(yaw), cos(yaw), valid]
        agent_yaw = torch.zeros_like(agent_trajs).to(device=self.device)
        # print(agent_yaw.shape, agent_ddqvels.shape)
        agent_yaw[..., 2] = agent_vels[..., 2]
        yaw = torch.atan2(agent_vels[..., 1], agent_vels[..., 0])
        agent_yaw[..., 0] = torch.sin(yaw)
        agent_yaw[..., 1] = torch.cos(yaw)

        x = torch.cat(
            [
                agent_trajs,  # 3
                agent_type.unsqueeze(0).repeat(N, 1, 1),  # N
                agent_vels,  # 3
                agent_yaw,  # 3
            ],
            dim=-1,
        )

        py_batched = Batch.from_data_list(
            [Data(*d) for d in zip(x, edge_index, edge_weight)]
        )
        return py_batched

    def forward(self, batch):
        batch_size = len(batch["obj_trajs"])
        batched_trajs, batch_obj_types, batched_obj_vels = (
            batch["obj_trajs"],
            batch["obj_types"],
            batch["obj_vels"],
        )


        st_feats = []
        if "graph" in self.config.st_feats:
            graphy_feats = []
            for objs_traj, objs_type, objs_vel in zip(
                batched_trajs, batch_obj_types, batched_obj_vels
            ):
                # iterate over batch
                graphs = self.build_agents_st_graphs(
                    objs_traj.transpose(0, 1), objs_type, objs_vel.transpose(0, 1)
                )

                x = self.conv1(graphs.x, graphs.edge_index, graphs.edge_attr)  # type: ignore
                x = self.conv2(x, graphs.edge_index, graphs.edge_attr)  # type: ignore

                if self.config.gnn_reduction != "ego_only":
                    x = scatter(
                        x, graphs.batch, dim=0, reduce=self.config.gnn_reduction
                    )  # type: ignore
                else:
                    # only use ego node, this may have bugs
                    # NOTE make sure ego is node with index 0 in every graph
                    _, node_indices = unique_with_index(graphs.batch, dim=0)
                    x = x[node_indices]

                graphy_feats.append(x)
            graphy_feats = torch.stack(graphy_feats)  # B T F
            st_feats.append(graphy_feats)

        # st_feats = graphy_feats
        # print(batched_hint.shape, graphy_feats.shape, ego_trajs.shape)
        if "ego" in self.config.st_feats:
            ego_trajs = torch.stack([trajs[0] for trajs in batched_trajs])
            st_feats.append(ego_trajs)

        if "hint" in self.config.st_feats:
            batched_hint = torch.stack(batch["hint"], dim=0).to(device=self.device)
            st_feats.append(batched_hint)

        st_feats = torch.cat(st_feats, dim=-1)
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
