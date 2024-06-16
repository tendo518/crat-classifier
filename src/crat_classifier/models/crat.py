from dataclasses import dataclass

import lightning.pytorch as pl
import numpy as np
import torch
from scipy import sparse
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import conv
from torch_geometric.utils import from_scipy_sparse_matrix

from .base_model import BaseModelConfig


class CratTrajClassifier(pl.LightningModule):
    @dataclass
    class ModelConfig(BaseModelConfig):
        latent_size: int = 64
        dp_ratio: float = 0.3

    def __init__(self, model_config: ModelConfig):
        super(CratTrajClassifier, self).__init__()

        self.config = model_config

        self.encoder_lstm = EncoderLstm(self.config.latent_size)
        self.agent_gnn = AgentGnn(self.config.latent_size)
        self.multihead_self_attention = MultiheadSelfAttention(
            self.config.latent_size, self.config.dp_ratio
        )
        self.classifier = ClassificationNet(
            self.config.latent_size,
            self.config.num_preds,
            self.config.num_classes,
            self.config.dp_ratio,
        )
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, batch):
        displ, centers = batch["displ"], batch["centers"]

        # rotation, origin = batch["rotation"], batch["origin"]

        # Extract the number of agents in each sample of the current batch
        agents_per_sample = [x.shape[0] for x in displ]

        # Convert the list of tensors to tensors
        displ = torch.cat(displ, dim=0).to(device=self.device)
        centers = torch.cat(centers, dim=0).to(device=self.device)

        out_encoder_lstm = self.encoder_lstm(displ, agents_per_sample)
        out_agent_gnn = self.agent_gnn(out_encoder_lstm, centers, agents_per_sample)
        out_self_attention = self.multihead_self_attention(
            out_agent_gnn, agents_per_sample
        )
        out_self_attention = torch.stack([x[0] for x in out_self_attention])
        out = self.classifier(out_self_attention)
        return out


class EncoderLstm(nn.Module):
    def __init__(self, latent_size, input_size=4, num_layers=1):
        super(EncoderLstm, self).__init__()
        self.input_size = input_size
        self.hidden_size = latent_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
        )

    def forward(self, lstm_in, agents_per_sample):
        # lstm_in are all agents over all samples in the current batch
        # Format for LSTM has to be has to be (batch_size, timeseries_length, latent_size), because batch_first=True

        # Initialize the hidden state.
        # lstm_in.shape[0] corresponds to the number of all agents in the current batch
        lstm_hidden_state = torch.randn(
            self.num_layers, lstm_in.shape[0], self.hidden_size, device=lstm_in.device
        )
        lstm_cell_state = torch.randn(
            self.num_layers, lstm_in.shape[0], self.hidden_size, device=lstm_in.device
        )
        lstm_hidden = (lstm_hidden_state, lstm_cell_state)

        lstm_out, lstm_hidden = self.lstm(lstm_in, lstm_hidden)

        # lstm_out is the hidden state over all time steps from the last LSTM layer
        # In this case, only the features of the last time step are used
        return lstm_out[:, -1, :]


class DecoderLstm(nn.Module):
    pass


class AgentGnn(nn.Module):
    def __init__(self, latent_size):
        super(AgentGnn, self).__init__()
        self.latent_size = latent_size

        self.gcn1 = conv.CGConv(self.latent_size, dim=2, batch_norm=True, bias=False)
        self.gcn2 = conv.CGConv(self.latent_size, dim=2, batch_norm=True, bias=False)

    def forward(self, gnn_in, centers, agents_per_sample):
        # gnn_in is a batch and has the shape (batch_size, number_of_agents, latent_size)

        x, edge_index = (
            gnn_in,
            self.build_fully_connected_edge_idx(agents_per_sample).to(gnn_in.device),
        )
        edge_attr = self.build_edge_attr(edge_index, centers).to(gnn_in.device)

        x = F.relu(self.gcn1(x, edge_index, edge_attr))
        gnn_out = F.relu(self.gcn2(x, edge_index, edge_attr))

        return gnn_out

    def build_fully_connected_edge_idx(self, agents_per_sample):
        edge_index = []

        # In the for loop one subgraph is built (no self edges!)
        # The subgraph gets offsetted and the full graph over all samples in the batch
        # gets appended with the offsetted subgrah
        offset = 0
        for i in range(len(agents_per_sample)):
            num_nodes = agents_per_sample[i]

            adj_matrix = torch.ones((num_nodes, num_nodes))
            adj_matrix = adj_matrix.fill_diagonal_(0)

            sparse_matrix = sparse.csr_matrix(adj_matrix.numpy())
            edge_index_subgraph, _ = from_scipy_sparse_matrix(sparse_matrix)

            # Offset the list
            edge_index_subgraph = torch.Tensor(np.asarray(edge_index_subgraph) + offset)
            offset += agents_per_sample[i]

            edge_index.append(edge_index_subgraph)

        # Concat the single subgraphs into one
        edge_index = torch.LongTensor(np.column_stack(edge_index))
        return edge_index

    def build_edge_attr(self, edge_index, data):
        edge_attr = torch.zeros((edge_index.shape[-1], 2), dtype=torch.float)

        rows, cols = edge_index
        # goal - origin
        edge_attr = data[cols] - data[rows]

        return edge_attr


class MultiheadSelfAttention(nn.Module):
    def __init__(self, latent_size, dp_ratio):
        super(MultiheadSelfAttention, self).__init__()

        self.latent_size = latent_size
        self.dp_ratio = dp_ratio

        self.multihead_attention = nn.MultiheadAttention(
            self.latent_size, 4, dropout=self.dp_ratio
        )

    def forward(self, att_in, agents_per_sample):
        att_out_batch = []

        # Upper path is faster for multiple samples in the batch and vice versa
        if len(agents_per_sample) > 1:
            max_agents = max(agents_per_sample)

            padded_att_in = torch.zeros(
                (len(agents_per_sample), max_agents, self.latent_size),
                device=att_in[0].device,
            )
            mask = torch.arange(max_agents) < torch.tensor(agents_per_sample)[:, None]

            padded_att_in[mask] = att_in

            mask_inverted = ~mask
            mask_inverted = mask_inverted.to(att_in.device)

            padded_att_in_swapped = torch.swapaxes(padded_att_in, 0, 1)

            padded_att_in_swapped, _ = self.multihead_attention(
                padded_att_in_swapped,
                padded_att_in_swapped,
                padded_att_in_swapped,
                key_padding_mask=mask_inverted,
            )

            padded_att_in_reswapped = torch.swapaxes(padded_att_in_swapped, 0, 1)

            att_out_batch = [
                x[0 : agents_per_sample[i]]
                for i, x in enumerate(padded_att_in_reswapped)
            ]
        else:
            att_in = torch.split(att_in, agents_per_sample)
            for i, sample in enumerate(att_in):
                # Add the batch dimension (this has to be the second dimension, because attention requires it)
                att_in_formatted = sample.unsqueeze(1)
                att_out, weights = self.multihead_attention(
                    att_in_formatted, att_in_formatted, att_in_formatted
                )

                # Remove the "1" batch dimension
                att_out = att_out.squeeze()
                att_out_batch.append(att_out)

        return att_out_batch


class ClassificationNet(nn.Module):
    def __init__(self, input_size, num_preds, num_class, dp_ratio):
        super(ClassificationNet, self).__init__()
        self.num_preds = num_preds
        self.num_class = num_class
        self.linear1 = nn.Linear(input_size, 128, bias=False)
        self.bn1 = nn.BatchNorm1d(128)
        self.dropout1 = nn.Dropout(dp_ratio)
        # self.linear2 = nn.Linear(128, out_features=256, bias=False)
        # self.bn2 = nn.BatchNorm1d(256)
        # self.dropout2 = nn.Dropout(self.args.dp_ratio)
        self.linear2 = nn.Linear(128, num_preds * num_class, bias=True)
        # self.bn3 = nn.BatchNorm1d(256)
        # self.linear4 = nn.Linear(256, args.num_preds * args.num_cls)

    def forward(self, classifier_in):
        out = self.bn1(self.linear1(classifier_in))
        out = F.relu(out)
        out = self.dropout1(out)
        # out = self.bn2(self.linear2(out))
        # out = F.relu(out)
        # out = self.dropout2(out)
        out = self.linear2(out)
        # out = F.relu(out)
        # out = self.linear4(out)
        # out = F.relu(out)
        # out = out.reshape(-1, self.args.num_cls)
        # out = nn.Softmax(dim=-1)(out)
        out = out.reshape(-1, self.num_preds, self.num_class)
        return out
