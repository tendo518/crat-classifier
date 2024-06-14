import lightning.pytorch as pl
import torch
from torch import nn, optim
from torch.nn import functional as F

from crat_classifier.configs import ModelConfig, OptimizerConfig
from crat_classifier.models import get_model


class Classifier(pl.LightningModule):
    def __init__(self, model_config: ModelConfig, optimizer_config: OptimizerConfig):
        super(Classifier, self).__init__()

        self.save_hyperparameters()

        self.model_config = model_config
        self.optimizer_config = optimizer_config

        self.model = get_model(model_config)
        self.ce_loss = nn.CrossEntropyLoss()

    def configure_optimizers(self):
        if self.optimizer_config.optimizer == "adam":
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=self.optimizer_config.learning_rate,
                weight_decay=self.optimizer_config.weight_decay,
            )
        elif self.optimizer_config.optimizer == "adamw":
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.optimizer_config.learning_rate,
                weight_decay=self.optimizer_config.weight_decay,
            )
        else:
            raise NotImplementedError

        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer,
            T_max=self.trainer.max_epochs // 4,  # type: ignore
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }

    def forward(self, batch):
        return self.model(batch)

    def prediction_loss(self, batched_preds, batched_gts):
        batched_preds = torch.permute(batched_preds, [0, 2, 1])
        loss = self.ce_loss(batched_preds, batched_gts)
        return loss

    def training_step(self, train_batch, batch_idx):
        batch_size: int = len(train_batch["gt"])

        out = self.forward(train_batch)
        batched_gts: torch.Tensor = torch.stack(train_batch["gt"])
        loss = self.prediction_loss(out, batched_gts)
        self.log(
            "train/loss",
            loss / batch_size,
            on_step=True,
            prog_bar=True,
            batch_size=batch_size,
        )
        self.log(
            "train/acc",
            torch.mean((F.softmax(out, dim=-1).argmax(dim=-1) == batched_gts).float()),
            on_step=True,
            prog_bar=True,
            batch_size=batch_size,
        )
        return loss

    def validation_step(self, val_batch, batch_idx):
        batch_size: int = len(val_batch["gt"])
        out = self.forward(val_batch)
        batched_gts = torch.stack(val_batch["gt"])

        loss = self.prediction_loss(out, batched_gts)
        # print(torch.argmax(out, dim=2)[0].cpu().numpy())
        # print(val_batch["gt"][0].cpu().numpy())
        pred_cls = F.softmax(out, dim=-1).argmax(dim=-1)
        self.log(
            "val/acc",
            torch.mean((pred_cls == batched_gts).float()),
            on_epoch=True,
            batch_size=batch_size,
        )
        self.log(
            "val/loss",
            loss / len(out),
            on_step=True,
            prog_bar=True,
            batch_size=batch_size,
        )
        self.log(
            "val/whatevereverthingisinlane",
            torch.mean((batched_gts == 0).float()),
            on_epoch=True,
            batch_size=batch_size,
        )
        self.log(
            "val/allyouneedisinlane",
            torch.mean((pred_cls == 0).float()),
            on_epoch=True,
            batch_size=batch_size,
        )

        # Extract target agent only
        pred = [x[0].detach().cpu().numpy() for x in out]
        gt = [x[0].detach().cpu().numpy() for x in val_batch["gt"]]
        return pred, gt
