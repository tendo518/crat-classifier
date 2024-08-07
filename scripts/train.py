import lightning.pytorch as pl
import torch
import tyro
from crat_classifier.configs import Config
from crat_classifier.dataset.dataset_utils import collate_fn_dict
from crat_classifier.dataset.suscape_csv_dataset import CSVDataset
from crat_classifier.trainer import Classifier
from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    EarlyStopping,
)
from torch.utils.data import DataLoader

torch.set_float32_matmul_precision("high")
torch.backends.cudnn.benchmark = True


def main(configs: Config):
    if configs.experiment.seed is not None:
        seed_everything(configs.experiment.seed)

    train_dataset = CSVDataset(configs.experiment.train_split)
    train_loader = DataLoader(
        train_dataset,
        batch_size=configs.experiment.batch_size,
        num_workers=configs.experiment.num_workers,
        collate_fn=collate_fn_dict,
        pin_memory=True,
        drop_last=True,
        shuffle=True,
    )
    val_dataset = CSVDataset(configs.experiment.val_split)
    val_loader = DataLoader(
        val_dataset,
        batch_size=configs.experiment.batch_size,
        num_workers=configs.experiment.num_workers,
        collate_fn=collate_fn_dict,
        drop_last=True,
        pin_memory=True,
    )

    callbacks = []
    callbacks.append(ModelCheckpoint(monitor="val/acc", save_top_k=5, mode="max"))
    callbacks.append(LearningRateMonitor(logging_interval="step"))
    if configs.experiment.early_stop:
        callbacks.append(EarlyStopping(monitor="val/acc", mode="max", patience=50))

    model = Classifier(configs.model, configs.optimizer)
    trainer = pl.Trainer(
        default_root_dir=configs.experiment.output_root,
        callbacks=callbacks,
        accelerator="gpu",
        devices=configs.experiment.gpus,
        max_epochs=configs.experiment.num_epochs,
        log_every_n_steps=5,
        check_val_every_n_epoch=5,
    )

    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    main(tyro.cli(Config))
