from dataclasses import dataclass

import lightning.pytorch as pl
import torch
from torch import nn
from torch.nn import functional as F

from crat_classifier.models.base_model import BaseModelConfig


class EncoderDecoderClassifier(pl.LightningModule):
    @dataclass
    class ModelConfig(BaseModelConfig):
        latent_size: int = 64
        num_preds: int = 40
        dp_ratio: float = 0.3

    def __init__(self, config: ModelConfig):
        super(EncoderDecoderClassifier, self).__init__()

        self.encoder = EncoderRNN(
            4, hidden_size=config.latent_size, dropout_p=config.dp_ratio
        )
        self.decoder = DecoderRNN(
            hidden_size=config.latent_size, output_size=config.num_classes
        )

        self.loss = nn.CrossEntropyLoss()

    def forward(self, batch):
        displ, centers = batch["displ"], batch["centers"]

        # rotation, origin = batch["rotation"], batch["origin"]

        # Extract the number of agents in each sample of the current batch
        agents_per_sample = [x.shape[0] for x in displ]

        # Convert the list of tensors to tensors
        displ = torch.cat(displ, dim=0).to(
            device=self.device
        )  # (N*agents_per_sample) 39 3
        centers = torch.cat(centers, dim=0).to(
            device=self.device
        )  # (N*agents_per_sample) 3

        encoder_out = self.encoder(displ)
        self.decoder.init_state(encoder_out)
        decoder_out = self.decoder(encoder_out)
        return decoder_out

    def training_step(self):
        pass

    def validation_step(self):
        pass


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_p=0.1):
        super(EncoderRNN, self).__init__()

        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, input):
        output, hidden = self.gru(input)
        return output, hidden


class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, encoder_outputs, encoder_hidden, target_tensor=None):
        batch_size = encoder_outputs.size(0)
        decoder_input = torch.empty(
            batch_size, 1, dtype=torch.long, device=device
        ).fill_(SOS_token)
        decoder_hidden = encoder_hidden
        decoder_outputs = []

        for i in range(MAX_LENGTH):
            decoder_output, decoder_hidden = self.forward_step(
                decoder_input, decoder_hidden
            )
            decoder_outputs.append(decoder_output)

            if target_tensor is not None:
                # Teacher forcing: Feed the target as the next input
                decoder_input = target_tensor[:, i].unsqueeze(1)  # Teacher forcing
            else:
                # Without teacher forcing: use its own predictions as the next input
                _, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze(
                    -1
                ).detach()  # detach from history as input

        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)
        return (
            decoder_outputs,
            decoder_hidden,
            None,
        )  # We return `None` for consistency in the training loop

    def forward_step(self, input, hidden):
        output = self.embedding(input)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.out(output)
        return output, hidden
